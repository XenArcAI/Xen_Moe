#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include <jaxlib/cuda/cuda_gpu_kernel_helpers.h>
#include <jaxlib/cuda/cuda_workspace_allocator.h>
#include <xla/client/xla_builder.h>
#include <xla/client/xla_computation.h>
#include <xla/client/lib/constants.h>
#include <xla/client/lib/matrix.h>
#include <xla/client/lib/prng.h>
#include <xla/client/lib/slicing.h>
#include <xla/client/xla_builder.h>
#include <xla/client/xla_computation.h>
#include <xla/literal.h>
#include <xla/shape.h>
#include <xla/shape_util.h>
#include <xla/status.h>
#include <xla/statusor.h>
#include <xla/util.h>
#include <xla/xla_data.pb.h>
#include <absl/status/status.h>

namespace xen_moe {
namespace tpu_kernels {

// Precision types for TPU communication
enum class Precision {
    FLOAT32,
    BFLOAT16,
    INT8
};

// Buffer configuration with precision support
struct TPUBufferConfig {
    int64_t buffer_size;      // Size of the communication buffer
    int num_channels;         // Number of communication channels
    bool use_ici;            // Whether to use ICI (intra-node) interconnect
    Precision precision;     // Data precision for communication
    int num_workers;         // Number of TPU workers
    int num_experts;         // Number of experts
    int hidden_dim;          // Hidden dimension size
    int batch_size;          // Batch size
    float scale_factor;      // Scale factor for INT8 quantization
};

// Device configuration
struct TPUDeviceConfig {
    int device_id;           // TPU device ID
    int num_cores;           // Number of TPU cores
    int ici_bandwidth;       // ICI bandwidth in GB/s
    int dci_bandwidth;       // DCI bandwidth in GB/s
    bool is_host;           // Whether this is a host device
    Precision default_precision; // Default precision for communication
};

// Communication buffer with precision support
struct MoEBuffer {
    void* data;              // Buffer data
    int64_t size;           // Buffer size
    Precision precision;    // Data precision
    bool is_ici;           // Whether using ICI
    float scale_factor;    // Scale factor for INT8
};

class TPUDeviceManager {
public:
    static TPUDeviceManager& GetInstance() {
        static TPUDeviceManager instance;
        return instance;
    }

    absl::Status Initialize(const TPUDeviceConfig& config) {
        config_ = config;
        return InitializeDevices();
    }

    absl::Status AllocateBuffer(MoEBuffer* buffer) {
        // Calculate size based on precision
        size_t element_size = GetElementSize(buffer->precision);
        buffer->size = buffer->size * element_size;
        
        // Allocate memory on TPU device
        PJRT_Device* device = GetDevice(buffer->num_workers);
        if (!device) {
            return absl::InternalError("Failed to get TPU device");
        }

        PJRT_Buffer* pjrt_buffer = nullptr;
        PJRT_Error* error = PJRT_Device_MemoryAllocate(
            device, buffer->size, &pjrt_buffer);
        
        if (error) {
            return absl::InternalError("Failed to allocate TPU memory");
        }

        buffer->data = pjrt_buffer;
        return absl::OkStatus();
    }

    absl::Status FreeBuffer(MoEBuffer* buffer) {
        if (!buffer->data) {
            return absl::OkStatus();
        }

        PJRT_Buffer* pjrt_buffer = static_cast<PJRT_Buffer*>(buffer->data);
        PJRT_Error* error = PJRT_Buffer_Delete(pjrt_buffer);
        
        if (error) {
            return absl::InternalError("Failed to free TPU memory");
        }

        buffer->data = nullptr;
        return absl::OkStatus();
    }

private:
    TPUDeviceManager() = default;
    ~TPUDeviceManager() = default;

    size_t GetElementSize(Precision precision) {
        switch (precision) {
            case Precision::FLOAT32:
                return sizeof(float);
            case Precision::BFLOAT16:
                return sizeof(uint16_t);
            case Precision::INT8:
                return sizeof(int8_t);
            default:
                return sizeof(float);
        }
    }

    // ... existing private methods ...
};

class MoEAllToAll {
public:
    static absl::Status Dispatch(const MoEBuffer* input_buffer,
                               MoEBuffer* output_buffer,
                               const std::vector<int>& expert_assignments) {
        // Validate input
        if (!input_buffer || !output_buffer) {
            return absl::InvalidArgumentError("Invalid buffer pointers");
        }

        // Get device manager
        auto& device_manager = TPUDeviceManager::GetInstance();

        // Allocate output buffer if needed
        if (!output_buffer->data) {
            auto status = device_manager.AllocateBuffer(output_buffer);
            if (!status.ok()) {
                return status;
            }
        }

        // Convert input data to target precision if needed
        if (input_buffer->precision != output_buffer->precision) {
            auto status = ConvertPrecision(input_buffer, output_buffer);
            if (!status.ok()) {
                return status;
            }
        }

        // Perform all-to-all communication
        // ... existing all-to-all implementation ...
    }

private:
    static absl::Status ConvertPrecision(const MoEBuffer* input,
                                       MoEBuffer* output) {
        switch (output->precision) {
            case Precision::BFLOAT16:
                return ConvertToBF16(input, output);
            case Precision::INT8:
                return ConvertToInt8(input, output);
            default:
                return absl::OkStatus();
        }
    }

    static absl::Status ConvertToBF16(const MoEBuffer* input,
                                    MoEBuffer* output) {
        // Convert float32 to bfloat16
        float* input_data = static_cast<float*>(input->data);
        uint16_t* output_data = static_cast<uint16_t*>(output->data);
        
        for (int i = 0; i < input->size / sizeof(float); ++i) {
            // Extract exponent and mantissa
            uint32_t bits = *reinterpret_cast<uint32_t*>(&input_data[i]);
            uint32_t sign = bits & 0x80000000;
            uint32_t exp = (bits >> 23) & 0xFF;
            uint32_t mant = bits & 0x7FFFFF;
            
            // Convert to bfloat16
            uint16_t bf16 = (sign >> 16) | ((exp & 0xFF) << 7) | (mant >> 16);
            output_data[i] = bf16;
        }
        
        return absl::OkStatus();
    }

    static absl::Status ConvertToInt8(const MoEBuffer* input,
                                    MoEBuffer* output) {
        // Convert float32 to int8 with scaling
        float* input_data = static_cast<float*>(input->data);
        int8_t* output_data = static_cast<int8_t*>(output->data);
        
        // Find max absolute value for scaling
        float max_abs = 0.0f;
        for (int i = 0; i < input->size / sizeof(float); ++i) {
            max_abs = std::max(max_abs, std::abs(input_data[i]));
        }
        
        // Calculate scale factor
        float scale = 127.0f / max_abs;
        output->scale_factor = scale;
        
        // Convert with scaling
        for (int i = 0; i < input->size / sizeof(float); ++i) {
            output_data[i] = static_cast<int8_t>(input_data[i] * scale);
        }
        
        return absl::OkStatus();
    }
};

} // namespace tpu_kernels
} // namespace xen_moe 