#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace xen_moe {

// Forward declarations
class Buffer;
class Config;
class EventHandle;

// Device type enum
enum class DeviceType {
    GPU,
    TPU
};

// Backend configuration
struct BackendConfig {
    DeviceType device_type;
    int num_workers;
    int num_experts;
    int hidden_dim;
    int batch_size;
    bool use_ici;  // For TPU: whether to use ICI interconnect
    bool low_latency_mode;  // For GPU: whether to use low-latency mode
};

// Abstract base class for backend implementations
class Backend {
public:
    virtual ~Backend() = default;

    // Initialize the backend
    virtual void initialize() = 0;

    // Clean up resources
    virtual void cleanup() = 0;

    // Dispatch tokens to experts
    virtual void dispatch(const void* input_data,
                         void* output_data,
                         const int64_t* expert_assignments,
                         int num_tokens,
                         int num_experts,
                         int hidden_dim) = 0;

    // Combine expert outputs
    virtual void combine(const void* input_data,
                        void* output_data,
                        const int64_t* expert_assignments,
                        const float* expert_weights,
                        int num_tokens,
                        int num_experts,
                        int hidden_dim) = 0;

    // Get optimal buffer configuration
    virtual void get_optimal_config(int batch_size,
                                  int hidden_dim,
                                  int num_experts,
                                  int num_workers,
                                  void* config) = 0;

    // Get device type
    virtual DeviceType get_device_type() const = 0;
};

// GPU backend implementation
class GPUBackend : public Backend {
public:
    GPUBackend(const BackendConfig& config);
    ~GPUBackend() override;

    void initialize() override;
    void cleanup() override;
    void dispatch(const void* input_data,
                 void* output_data,
                 const int64_t* expert_assignments,
                 int num_tokens,
                 int num_experts,
                 int hidden_dim) override;
    void combine(const void* input_data,
                void* output_data,
                const int64_t* expert_assignments,
                const float* expert_weights,
                int num_tokens,
                int num_experts,
                int hidden_dim) override;
    void get_optimal_config(int batch_size,
                           int hidden_dim,
                           int num_experts,
                           int num_workers,
                           void* config) override;
    DeviceType get_device_type() const override { return DeviceType::GPU; }

private:
    BackendConfig config_;
    std::unique_ptr<Buffer> buffer_;
};

// TPU backend implementation
class TPUBackend : public Backend {
public:
    TPUBackend(const BackendConfig& config);
    ~TPUBackend() override;

    void initialize() override;
    void cleanup() override;
    void dispatch(const void* input_data,
                 void* output_data,
                 const int64_t* expert_assignments,
                 int num_tokens,
                 int num_experts,
                 int hidden_dim) override;
    void combine(const void* input_data,
                void* output_data,
                const int64_t* expert_assignments,
                const float* expert_weights,
                int num_tokens,
                int num_experts,
                int hidden_dim) override;
    void get_optimal_config(int batch_size,
                           int hidden_dim,
                           int num_experts,
                           int num_workers,
                           void* config) override;
    DeviceType get_device_type() const override { return DeviceType::TPU; }

private:
    BackendConfig config_;
    std::unique_ptr<tpu::MoEAllToAll> all_to_all_;
    std::unique_ptr<tpu::TPUDeviceManager> device_manager_;
};

// Factory function to create backend
std::unique_ptr<Backend> create_backend(const BackendConfig& config);

// Helper function to get device type from string
DeviceType get_device_type_from_string(const std::string& device_type);

} // namespace xen_moe 