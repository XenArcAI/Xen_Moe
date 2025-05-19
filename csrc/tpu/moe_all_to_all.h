#ifndef XEN_MOE_TPU_MOE_ALL_TO_ALL_H_
#define XEN_MOE_TPU_MOE_ALL_TO_ALL_H_

#include <memory>
#include <vector>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xen_moe {
namespace tpu {

// TPU device configuration
struct TPUDeviceConfig {
  int num_cores;
  int num_devices;
  int num_hosts;
  int num_workers;
  bool use_ici;  // Use ICI (intra-node) interconnect
  bool use_dci;  // Use DCI (inter-node) interconnect
  int ici_bandwidth;  // ICI bandwidth in GB/s
  int dci_bandwidth;  // DCI bandwidth in GB/s
};

// MoE communication buffer
struct MoEBuffer {
  void* data;
  size_t size;
  int num_tokens;
  int hidden_dim;
  int num_experts;
  int num_workers;
  bool is_dispatch;  // true for dispatch, false for combine
};

// TPU device manager
class TPUDeviceManager {
public:
  static TPUDeviceManager& GetInstance();
  absl::Status Initialize(const TPUDeviceConfig& config);
  absl::Status AllocateBuffer(MoEBuffer* buffer);
  absl::Status FreeBuffer(MoEBuffer* buffer);

private:
  TPUDeviceManager() = default;
  ~TPUDeviceManager() = default;
  TPUDeviceManager(const TPUDeviceManager&) = delete;
  TPUDeviceManager& operator=(const TPUDeviceManager&) = delete;

  absl::Status InitializeDevices();
  void* GetDevice(int worker_id);
  void* GetTPUDevice(void* client, int device_id);

  TPUDeviceConfig config_;
  std::vector<void*> devices_;
};

// MoE all-to-all communication implementation
class MoEAllToAll {
public:
  static absl::Status Dispatch(const MoEBuffer* input_buffer,
                             MoEBuffer* output_buffer,
                             const std::vector<int>& expert_assignments);

  static absl::Status Combine(const MoEBuffer* input_buffer,
                            MoEBuffer* output_buffer,
                            const std::vector<int>& expert_assignments);

private:
  MoEAllToAll() = default;
  ~MoEAllToAll() = default;
  MoEAllToAll(const MoEAllToAll&) = delete;
  MoEAllToAll& operator=(const MoEAllToAll&) = delete;
};

}  // namespace tpu
}  // namespace xen_moe

#endif  // XEN_MOE_TPU_MOE_ALL_TO_ALL_H_ 