#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace xen_moe {

// Auto-tuning parameters for different batch sizes
struct AutoTuneConfig {
    // Buffer sizes for different batch size ranges
    struct BufferSizes {
        int64_t nvl_bytes;  // NVLink buffer size
        int64_t rdma_bytes; // RDMA buffer size
        int num_channels;   // Number of communication channels
    };

    // Kernel configurations
    struct KernelConfig {
        int num_warps_per_group;     // Number of warps per group
        int num_warp_groups;         // Number of warp groups
        int num_forwarder_warps;     // Number of forwarder warps
        int num_rdma_receivers;      // Number of RDMA receivers
        bool use_fp8;                // Whether to use FP8 for communication
        bool zero_copy;              // Whether to use zero-copy for RDMA
    };

    // Get optimal buffer sizes based on batch size
    static BufferSizes get_buffer_sizes(int batch_size, int hidden_dim, int num_experts) {
        BufferSizes sizes;
        
        // Calculate base sizes
        int64_t token_size = batch_size * hidden_dim * sizeof(nv_bfloat16);
        
        // Auto-tune based on batch size ranges
        if (batch_size <= 128) {
            // Small batch size: prioritize latency
            sizes.nvl_bytes = token_size * 2;  // 2x for double buffering
            sizes.rdma_bytes = token_size * 4; // 4x for RDMA buffering
            sizes.num_channels = 2;            // Fewer channels for small batches
        } else if (batch_size <= 1024) {
            // Medium batch size: balance latency and throughput
            sizes.nvl_bytes = token_size * 4;
            sizes.rdma_bytes = token_size * 8;
            sizes.num_channels = 4;
        } else {
            // Large batch size: prioritize throughput
            sizes.nvl_bytes = token_size * 8;
            sizes.rdma_bytes = token_size * 16;
            sizes.num_channels = 8;
        }

        // Scale with number of experts
        float expert_scale = std::sqrt(static_cast<float>(num_experts) / 8.0f);
        sizes.nvl_bytes = static_cast<int64_t>(sizes.nvl_bytes * expert_scale);
        sizes.rdma_bytes = static_cast<int64_t>(sizes.rdma_bytes * expert_scale);

        return sizes;
    }

    // Get optimal kernel configuration based on batch size and device
    static KernelConfig get_kernel_config(int batch_size, int num_experts, cudaDeviceProp& prop) {
        KernelConfig config;
        
        // Determine if we should use FP8 based on device capability
        config.use_fp8 = prop.major >= 9;  // Hopper and newer
        
        // Auto-tune warp configuration based on batch size
        if (batch_size <= 128) {
            config.num_warps_per_group = 8;   // Fewer warps for small batches
            config.num_warp_groups = 2;       // Fewer groups for better latency
            config.num_forwarder_warps = 4;   // Fewer forwarders
            config.num_rdma_receivers = 8;    // Fewer receivers
            config.zero_copy = true;          // Enable zero-copy for small batches
        } else if (batch_size <= 1024) {
            config.num_warps_per_group = 10;
            config.num_warp_groups = 3;
            config.num_forwarder_warps = 8;
            config.num_rdma_receivers = 12;
            config.zero_copy = false;
        } else {
            config.num_warps_per_group = 12;
            config.num_warp_groups = 4;
            config.num_forwarder_warps = 12;
            config.num_rdma_receivers = 16;
            config.zero_copy = false;
        }

        // Scale with number of experts
        float expert_scale = std::sqrt(static_cast<float>(num_experts) / 8.0f);
        config.num_forwarder_warps = static_cast<int>(config.num_forwarder_warps * expert_scale);
        config.num_rdma_receivers = static_cast<int>(config.num_rdma_receivers * expert_scale);

        return config;
    }
};

} // namespace xen_moe 