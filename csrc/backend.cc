#include "backend.h"
#include "buffer.h"
#include "tpu_kernels/moe_all_to_all_tpu.h"
#include <stdexcept>

namespace xen_moe {

// GPUBackend implementation
GPUBackend::GPUBackend(const BackendConfig& config)
    : config_(config) {}

GPUBackend::~GPUBackend() {
    cleanup();
}

void GPUBackend::initialize() {
    // Initialize GPU buffer with optimal configuration
    void* gpu_config = nullptr;
    get_optimal_config(config_.batch_size,
                      config_.hidden_dim,
                      config_.num_experts,
                      config_.num_workers,
                      &gpu_config);

    buffer_ = std::make_unique<Buffer>(
        nullptr,  // Process group will be set by the user
        0,        // NVLink buffer size will be set by optimal config
        0,        // RDMA buffer size will be set by optimal config
        config_.low_latency_mode
    );
}

void GPUBackend::cleanup() {
    buffer_.reset();
}

void GPUBackend::dispatch(const void* input_data,
                         void* output_data,
                         const int64_t* expert_assignments,
                         int num_tokens,
                         int num_experts,
                         int hidden_dim) {
    if (!buffer_) {
        throw std::runtime_error("GPU backend not initialized");
    }

    // Convert input data to GPU buffer format
    // This would typically involve copying data to GPU memory
    // For now, we'll just pass through the pointers
    buffer_->dispatch(
        input_data,
        output_data,
        expert_assignments,
        num_tokens,
        num_experts,
        hidden_dim
    );
}

void GPUBackend::combine(const void* input_data,
                        void* output_data,
                        const int64_t* expert_assignments,
                        const float* expert_weights,
                        int num_tokens,
                        int num_experts,
                        int hidden_dim) {
    if (!buffer_) {
        throw std::runtime_error("GPU backend not initialized");
    }

    // Convert input data to GPU buffer format
    // This would typically involve copying data to GPU memory
    // For now, we'll just pass through the pointers
    buffer_->combine(
        input_data,
        output_data,
        expert_assignments,
        expert_weights,
        num_tokens,
        num_experts,
        hidden_dim
    );
}

void GPUBackend::get_optimal_config(int batch_size,
                                  int hidden_dim,
                                  int num_experts,
                                  int num_workers,
                                  void* config) {
    // Get optimal configuration for GPU
    // This would use the existing GPU auto-tuning logic
    // For now, we'll just set some default values
    if (config) {
        auto* gpu_config = static_cast<Config*>(config);
        gpu_config->num_sms = 20;
        gpu_config->num_warps = 32;
        gpu_config->num_threads = 1024;
    }
}

// TPUBackend implementation
TPUBackend::TPUBackend(const BackendConfig& config)
    : config_(config) {}

TPUBackend::~TPUBackend() {
    cleanup();
}

void TPUBackend::initialize() {
    // Initialize TPU device manager
    tpu::TPUDeviceConfig device_config;
    device_config.device_id = 0;  // Use first TPU device
    device_config.num_cores = 4;  // Assuming 4 cores per TPU
    device_config.ici_bandwidth = 200;  // 200 GB/s for ICI
    device_config.dci_bandwidth = 100;  // 100 GB/s for DCI
    device_config.is_host = true;

    device_manager_ = std::make_unique<tpu::TPUDeviceManager>(device_config);
    device_manager_->initialize();

    // Initialize TPU all-to-all with optimal configuration
    tpu::TPUBufferConfig buffer_config = tpu::MoEAllToAll::get_optimal_config(
        config_.batch_size,
        config_.hidden_dim,
        config_.num_experts,
        config_.num_workers,
        config_.use_ici
    );

    all_to_all_ = std::make_unique<tpu::MoEAllToAll>(device_config, buffer_config);
    all_to_all_->initialize();
}

void TPUBackend::cleanup() {
    all_to_all_.reset();
    device_manager_.reset();
}

void TPUBackend::dispatch(const void* input_data,
                         void* output_data,
                         const int64_t* expert_assignments,
                         int num_tokens,
                         int num_experts,
                         int hidden_dim) {
    if (!all_to_all_) {
        throw std::runtime_error("TPU backend not initialized");
    }

    all_to_all_->dispatch(
        input_data,
        output_data,
        expert_assignments,
        num_tokens,
        num_experts,
        hidden_dim
    );
}

void TPUBackend::combine(const void* input_data,
                        void* output_data,
                        const int64_t* expert_assignments,
                        const float* expert_weights,
                        int num_tokens,
                        int num_experts,
                        int hidden_dim) {
    if (!all_to_all_) {
        throw std::runtime_error("TPU backend not initialized");
    }

    all_to_all_->combine(
        input_data,
        output_data,
        expert_assignments,
        expert_weights,
        num_tokens,
        num_experts,
        hidden_dim
    );
}

void TPUBackend::get_optimal_config(int batch_size,
                                  int hidden_dim,
                                  int num_experts,
                                  int num_workers,
                                  void* config) {
    if (config) {
        auto* tpu_config = static_cast<tpu::TPUBufferConfig*>(config);
        *tpu_config = tpu::MoEAllToAll::get_optimal_config(
            batch_size,
            hidden_dim,
            num_experts,
            num_workers,
            config_.use_ici
        );
    }
}

// Factory function implementation
std::unique_ptr<Backend> create_backend(const BackendConfig& config) {
    switch (config.device_type) {
        case DeviceType::GPU:
            return std::make_unique<GPUBackend>(config);
        case DeviceType::TPU:
            return std::make_unique<TPUBackend>(config);
        default:
            throw std::runtime_error("Unsupported device type");
    }
}

// Helper function implementation
DeviceType get_device_type_from_string(const std::string& device_type) {
    if (device_type == "gpu") {
        return DeviceType::GPU;
    } else if (device_type == "tpu") {
        return DeviceType::TPU;
    } else {
        throw std::runtime_error("Unsupported device type: " + device_type);
    }
}

} // namespace xen_moe 