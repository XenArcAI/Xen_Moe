#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"
#include "auto_tune.cuh"

namespace xen_moe {

namespace internode_ll {

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure low-latency mode work fine)
    nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch(void* packed_recv_x, float* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // Clean up next buffer
    if (responsible_expert_idx < num_experts) {
        if (sub_warp_id == 0 and lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < num_next_clean_int; ++i)
                next_clean[i] = 0;
        }
        __syncwarp();
    }

    // Iterate over all tokens
    if (responsible_expert_idx < num_experts) {
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto global_expert_idx = dst_rank * num_local_experts + local_expert_idx;

        // Iterate over tokens
        for (int64_t token_idx = lane_id; token_idx < num_tokens; token_idx += 32) {
            // Check if token is assigned to this expert
            const auto token_expert_idx = topk_idx[token_idx * num_topk];
            if (token_expert_idx != global_expert_idx)
                continue;

            // Get slot index
            const auto slot_idx = atomic_add_release_global(atomic_counter_per_expert + global_expert_idx, 1);
            if (slot_idx >= num_max_dispatch_tokens_per_rank)
                continue;

            // Pack data
            if constexpr (kUseFP8) {
                // Use FP8 for communication on Hopper
                auto src_ptr = reinterpret_cast<const nv_bfloat16*>(x) + token_idx * kHidden;
                auto dst_ptr = reinterpret_cast<nv_bfloat16*>(packed_recv_x) + slot_idx * kHidden;
                #pragma unroll
                for (int i = 0; i < hidden_bf16_int4; ++i) {
                    const auto src_int4 = reinterpret_cast<const int4*>(src_ptr)[i];
                    reinterpret_cast<int4*>(dst_ptr)[i] = src_int4;
                }
            } else {
                // Use BF16 for communication on Ampere
                auto src_ptr = reinterpret_cast<const nv_bfloat16*>(x) + token_idx * kHidden;
                auto dst_ptr = reinterpret_cast<nv_bfloat16*>(packed_recv_x) + slot_idx * kHidden;
                #pragma unroll
                for (int i = 0; i < hidden_bf16_int4; ++i) {
                    const auto src_int4 = reinterpret_cast<const int4*>(src_ptr)[i];
                    reinterpret_cast<int4*>(dst_ptr)[i] = src_int4;
                }
            }

            // Pack metadata
            packed_recv_src_info[slot_idx] = token_idx;
            packed_recv_layout_range[slot_idx] = token_idx * num_topk;
            packed_recv_count[global_expert_idx] = slot_idx + 1;
        }
    }

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive and notify PCIe usage
    if (responsible_expert_idx < num_experts) {
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0)
            while (ld_acquire_sys_global(rdma_recv_count + responsible_expert_idx) == 0);
    }
    cg::this_grid().sync();

    // Copy data from RDMA buffer to local buffer
    if (responsible_expert_idx < num_experts) {
        const auto num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + responsible_expert_idx);
        if (num_recv_tokens > 0) {
            auto src_ptr = reinterpret_cast<const nv_bfloat16*>(rdma_recv_x) + responsible_expert_idx * num_max_dispatch_tokens_per_rank * kHidden;
            auto dst_ptr = reinterpret_cast<nv_bfloat16*>(packed_recv_x) + responsible_expert_idx * num_max_dispatch_tokens_per_rank * kHidden;
            #pragma unroll
            for (int i = 0; i < hidden_bf16_int4; ++i) {
                const auto src_int4 = reinterpret_cast<const int4*>(src_ptr)[i];
                reinterpret_cast<int4*>(dst_ptr)[i] = src_int4;
            }
        }
    }
}

void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks, bool use_fp8,
              void* workspace, cudaStream_t stream, int phases) {
    // Get device properties for auto-tuning
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get optimal configuration based on batch size
    auto config = AutoTuneConfig::get_kernel_config(num_tokens, num_experts, prop);
    
    // Override use_fp8 based on device capability
    use_fp8 = config.use_fp8;

    const auto num_warps = config.num_warp_groups * config.num_warps_per_group;
    const auto num_sms = cell_div(num_experts, config.num_warp_groups);

    // Check workspace
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(sizeof(int) * num_experts * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden) { \
    auto dispatch_func = use_fp8 ? \
        dispatch<true, kNumWarpGroups, kNumWarpsPerGroup, hidden> : \
        dispatch<false, kNumWarpGroups, kNumWarpsPerGroup, hidden>; \
    LAUNCH_KERNEL(&cfg, dispatch_func, \
                  packed_recv_x, packed_recv_x_scales, \
                  packed_recv_src_info, packed_recv_layout_range, \
                  packed_recv_count, \
                  rdma_recv_x, rdma_recv_count, rdma_x, \
                  x, topk_idx, \
                  atomic_counter_per_expert, atomic_finish_counter_per_expert, \
                  next_clean, num_next_clean_int, \
                  num_tokens, num_max_dispatch_tokens_per_rank, \
                  num_topk, num_experts, rank, num_ranks, \
                  phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put finishing flag
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            if (dst_rank != rank) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_recv_flag + global_expert_idx, 1, dst_rank, local_expert_idx);
            } else {
                st_na_release(rdma_recv_flag + global_expert_idx, 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive and notify PCIe usage
    if (responsible_expert_idx < num_experts) {
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0)
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
    }
    cg::this_grid().sync();

    // Reduce tokens with FP8 cast
    EP_DEVICE_ASSERT(num_topk <= 32 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                // Reduce
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             void* workspace, cudaStream_t stream,
             int phases, bool zero_copy) {
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
    constexpr int kNumMaxTopk = 9;

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              phases, zero_copy); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace xen_moe
