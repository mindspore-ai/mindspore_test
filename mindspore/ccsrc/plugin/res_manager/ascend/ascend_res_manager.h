/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_DEVICE_RES_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_DEVICE_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "acl/acl_rt.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "ir/tensor.h"
#include "utils/ms_context.h"
#include "include/backend/mem_reuse/dynamic_mem_pool.h"
#include "runtime/device/res_manager/hal_res_base.h"

namespace mindspore {
namespace device {
namespace ascend {
struct MemUceInfo {
  int device_id = 0;
  std::vector<aclrtMemUceInfo> info;
  size_t retSize = 0;
};

using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
class ASCEND_RES_MANAGER_EXPORT AscendResManager : public HalResBase {
 public:
  explicit AscendResManager(const ResKey &res_key) : HalResBase(res_key) { Initialize(); }
  ~AscendResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  void SetDeterministic() const;
  void SetDebugKernel() const;

  void SetAclDeterministic() const;

  CollectiveCommunicationLib *collective_comm_lib() const { return collective_comm_lib_; }
  std::shared_ptr<MemoryManager> mem_manager() { return mem_manager_; }
  std::shared_ptr<SwapManager> swap_manager() const { return swap_manager_; }

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                               uint32_t stream_id = kDefaultStreamIndex) const override;

  DeviceAddressPtr CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const override;
  DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                                       TypeId type_id, const std::string &device_name, uint32_t device_id,
                                       uint32_t stream_id) const override;

  bool LoadCollectiveCommLib() override;
  bool IsEnableVmm() const override;

  bool BindDeviceToCurrentThread(bool force_bind) const override;
  void *GetStream() const override { return AscendStreamMng::GetInstance().default_stream(); }
  void *GetCopyDataStream() const;

  // Relevant function to allocate and free device memory of raw ptr.
  bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const override;
  void *AllocateStaticMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override;
  void FreeMemory(DeviceAddress *const &address) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                       const std::vector<size_t> &keep_addr_sizes) const override;
  void DefragMemory() override;

  size_t GetMaxUsedMemorySize() const override;

  // Relevant function to manage memory statistics
  size_t GetTotalMemStatistics() const override;
  size_t GetTotalUsedMemStatistics() const override;
  size_t GetTotalIdleMemStatistics() const override;
  size_t GetTotalEagerFreeMemStatistics() const override;
  size_t GetUsedMemPeakStatistics() const override;
  size_t GetReservedMemPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> GetCommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const override;
  void ResetMaxMemoryReserved() override;
  void ResetMaxMemoryAllocated() override;

  size_t EmptyCache() override;

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override;
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override;

  bool CreateStream(size_t *stream_id) const override;
  bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void *GetStream(size_t stream_id) const override;
  void SetCurrentStreamId(size_t stream_id) override;
  size_t GetCurrentStreamId() const override;
  bool QueryStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id = 0) const override;
  bool SyncAllStreams() const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;
  std::pair<std::vector<size_t>, std::vector<size_t>> AllocDeviceMemoryForTensorList(
    const std::vector<tensor::BaseTensorPtr> &tensor_list, bool enable_mem_align) override;
  tensor::BaseTensorPtr GetSliceByTensorListIndexHandle(const std::vector<tensor::BaseTensorPtr> &tensor_list,
                                                        const std::vector<size_t> &before_padding_size,
                                                        const std::vector<size_t> &after_padding_size, size_t start,
                                                        size_t end) override;
  TensorPtr GetSliceByPaddingShapeHandle(const tensor::BaseTensorPtr &first_tensor, size_t start, size_t end) override;
  int ResetParams(const std::vector<tensor::TensorPtr> &params) const;

  DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait);
  DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) override;
  bool DestroyEvent(const DeviceEventPtr &event) override;
  bool DestroyAllEvents() override;

  bool single_op_multi_stream_enable() const override;
  void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) override;
  // Only used in graph_mode with MS_DISABLE_REF_MODE, delete it when delete MS_DISABLE_REF_MODEF
  void SetCPUMemManager();

  bool GetMemUceInfo(int32_t device_id) override;
  void UceMemRepair(int32_t device_id) override;
  void StopDevice(int32_t device_id) override;
  std::vector<std::pair<device::DeviceMemPtr, size_t>> GetMemUceAddr() override;
  bool AllocateForHete(DeviceAddress *const &address, mindspore::kernel::HeterogeneousInfoPtr hete_info) const;
  void FreeForHete(mindspore::kernel::HeterogeneousInfoPtr hete_info) const;

  // Override interface for multi stream event control.
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &input_event) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) override;

  bool SyncAllEvents() override;

  bool LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block = false) const override;

 private:
  MemUceInfo mem_uce_info_;
  std::mutex mem_uce_info_mutex_;
  bool initialized_ = false;
  // The collective communication library.
  std::function<CollectiveCommunicationLib *(void)> load_comm_lib_cb_;
  CollectiveCommunicationLib *collective_comm_lib_;

  std::shared_ptr<SwapManager> swap_manager_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
  DeviceEventPtrList device_events_{};
  std::mutex device_events_mutex_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_DEVICE_RES_MANAGER_H_
