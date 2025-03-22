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
#ifndef MINDSPORE_CCSR_RUNTIME_DEVICE_HAL_RES_BASE_H_
#define MINDSPORE_CCSR_RUNTIME_DEVICE_HAL_RES_BASE_H_
#include <map>
#include <type_traits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include "utils/log_adapter.h"
#include "include/backend/visible.h"
#include "common/device_address.h"
#include "ir/device_event.h"
#include "include/common/utils/anfalgo.h"

#include "runtime/collective/collective_communication_lib.h"
#include "runtime/collective/collective_comm_lib_loader.h"
#include "runtime/device/res_manager/auto_mem_offload.h"
#include "runtime/device/res_manager/swap_manager.h"
#include "runtime/device/res_manager/memory_manager.h"
#include "runtime/device/res_manager/utils/visible.h"
#include "runtime/device/res_manager/utils/utils.h"

namespace mindspore {
namespace device {
constexpr auto kDefaultStreamIndex = 0;
constexpr auto kWorldGroupStreamIndex = 1;

class RES_EXPORT HalResBase {
 public:
  explicit HalResBase(const ResKey &res_key) : res_key_(res_key) {}
  virtual ~HalResBase() = default;

  // Initialize the device resource manager.
  virtual void Initialize() {}

  // Destroy device resource manager and release device resource.
  virtual void Destroy() {}

  // Bind device to current thread to gain device control privileges
  // If force_bind is true, bind context to current thread every time;
  // Otherwise, only bind context to current thread for the first time.
  virtual bool BindDeviceToCurrentThread(bool force_bind) const { return true; }
  virtual void ResetStreamAndCtx() {}

  // Relevant function to allocate and free device memory of raw ptr.
  virtual void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const = 0;
  virtual void FreeMemory(void *ptr) const = 0;
  virtual void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                               const std::vector<size_t> &keep_addr_sizes) const = 0;
  virtual void DefragMemory() {}
  virtual bool IsEnableVmm() const { return false; }

  virtual void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return;
  }
  virtual void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return;
  }

  // Relevant function to allocate and free device memory of DeviceAddress.
  virtual bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return false;
  }
  virtual void FreeMemory(DeviceAddress *const &address) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return;
  }
  virtual size_t GetMaxUsedMemorySize() const { return 0; }

  // Relevant function to manage memory statistics
  virtual size_t GetTotalMemStatistics() const { return 0; }
  virtual size_t GetTotalUsedMemStatistics() const { return 0; }
  virtual size_t GetTotalIdleMemStatistics() const { return 0; }
  virtual size_t GetTotalEagerFreeMemStatistics() const { return 0; }
  virtual size_t GetUsedMemPeakStatistics() const { return 0; }
  virtual size_t GetReservedMemPeakStatistics() const { return 0; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const { return {}; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const { return {}; }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetCommonMemBlocksInfoStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const {
    return {};
  }
  virtual void ResetMaxMemoryReserved() {}
  virtual void ResetMaxMemoryAllocated() {}

  virtual size_t EmptyCache() { return -1L; }

  // Allocate host memory with raii and ref count
  virtual std::shared_ptr<void> AllocateHostMemory(size_t size) const {
    return std::shared_ptr<void>(::malloc(size), ::free);
  }
  // Allocate host memory for offload device memory.
  virtual void *AllocateOffloadMemory(size_t size) const;
  // Release host memory which was allocated by AllocateOffloadMemory to pool.
  // It will not be free to os.
  virtual void FreeOffloadMemory(void *ptr) const;

  virtual size_t GetAvailableMemSize() const { return 0; }

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  virtual std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                       uint32_t stream_id = kDefaultStreamIndex) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  // Create concrete device address according different device type using KernelTensor.
  virtual DeviceAddressPtr CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  virtual DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                               const Format &format, TypeId type_id, const std::string &device_name,
                                               uint32_t device_id, uint32_t stream_id) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  // Create a stream with assigning a stream id, the assigned stream id will be written to the parameter '*stream_id'.
  virtual bool CreateStream(size_t *stream_id) const {
    MS_LOG(WARNING) << "Unimplemented interface: 'CreateStream'.";
    *stream_id = kSizeZero;
    return false;
  }

  // Create a stream with priority.
  virtual bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
    *stream_id = kSizeZero;
    return false;
  }

  virtual size_t QueryStreamSize() const { return 0L; }
  virtual std::vector<uint32_t> GetStreamIds() const { return {}; }

  // If multi-stream used in pynative mode, other streams must be sync before the graph
  // is executed. Otherwise, out-of-order occurs. Therefore this flag is added.
  // This solution is a temporary solution, this flag will be removed after multi-stream is
  // supported in graph mode.
  virtual bool single_op_multi_stream_enable() const { return false; }
  virtual void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {}

  // Get the stream pointer by stream_id.
  virtual void *GetStream(size_t stream_id) const { return nullptr; }

  // Set currently using stream id.
  virtual void SetCurrentStreamId(size_t stream_id) { return; }

  // Get currently using stream id.
  virtual size_t GetCurrentStreamId() const { return kSizeZero; }

  virtual void *GetStream() const { return nullptr; }

  virtual size_t GetCommunicationStreamID() const { return kDefaultStreamIndex; }

  virtual size_t GetCommunicationStreamIDByGroup(const std::string &group) const { return GetCommunicationStreamID(); }

  // Destroy a stream bound to the input parameter "stream_id".
  virtual bool DestroyStream(size_t stream_id) const { return false; }

  // Query tasks' completion status of a stream.
  virtual bool QueryStream(size_t stream_id) const { return true; }

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // Using 'SyncStream' to block thread and wait for completing all tasks on specific stream.
  // Using 'SyncAllStream' to block thread and wait for completing all tasks on all streams.
  // Devices without stream could ignore the implementation of these function.
  // Since the current entry for creating streams is not unified, the implementation of the 'SyncStream' and
  // "SyncAllStreams" interfaces are implemented by subclasses.
  virtual bool SyncStream(size_t stream_id) const { return true; }

  virtual bool SyncAllStreams() const { return true; }

  virtual bool SyncNotDefaultStreams() const { return true; }

  // Return default stream id. Normally it's 0.
  virtual size_t DefaultStream() const { return 0; }

  // Create device event for runtime.
  virtual DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) { return nullptr; }

  // Create device event with flag.
  virtual DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api = true) {
    return nullptr;
  }

  // Destroy specified device event.
  virtual bool DestroyEvent(const DeviceEventPtr &event) { return false; }

  // Destroy all device events.
  virtual bool DestroyAllEvents() { return false; }

  // Detect stress.
  virtual int StressDetect() const { MS_LOG(EXCEPTION) << "Stress detection is not supported."; }
  // Send and receive parameters.
  virtual int SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank) const {
    MS_LOG(EXCEPTION) << "Send and receive parameters is not supported.";
  }

  // Clean tdt channel
  virtual int CleanTdtChannel() const { MS_LOG(EXCEPTION) << "Clean tdt channel is not supported."; }

  // Dynamically load collective communication library.
  // Currently, four types are supported: OpenMPI and self developed framework for CPU. NCCL for GPU. HCCL for Ascend.
  virtual bool LoadCollectiveCommLib() { return true; }

  // Return collective communication object for caller to access
  virtual CollectiveCommunicationLib *collective_comm_lib() const { return nullptr; }

  virtual std::shared_ptr<MemoryManager> mem_manager() { return nullptr; }

  virtual std::shared_ptr<SwapManager> swap_manager() const { return nullptr; }

  virtual std::pair<std::vector<size_t>, std::vector<size_t>> AllocDeviceMemoryForTensorList(
    const std::vector<tensor::BaseTensorPtr> &tensor_list, bool enable_mem_align) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  virtual tensor::BaseTensorPtr GetSliceByTensorListIndexHandle(const std::vector<tensor::BaseTensorPtr> &tensor_list,
                                                                const std::vector<size_t> &before_padding_size,
                                                                const std::vector<size_t> &after_padding_size,
                                                                size_t start, size_t end) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
  virtual tensor::TensorPtr GetSliceByPaddingShapeHandle(const tensor::BaseTensorPtr &first_tensor, size_t start,
                                                         size_t end) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
  virtual bool GetMemUceInfo(int32_t device_id) { return false; }
  virtual void UceMemRepair(int32_t device_id) { MS_LOG(EXCEPTION) << "Uce repair device is not supported."; }
  virtual void StopDevice(int32_t device_id) { MS_LOG(EXCEPTION) << "Uce stop device is not supported."; }
  virtual std::vector<std::pair<device::DeviceMemPtr, size_t>> GetMemUceAddr() { return {}; }

  virtual bool LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block = false) const {
    callback_func();
    return true;
  }

  // Interface for multi stream event control.
  virtual bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                           const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                           const DeviceEventPtr &input_event) {
    return false;
  }

  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
    return false;
  }

  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) { return false; }

  virtual bool SyncAllEvents() { return false; }

 protected:
  ResKey res_key_;
};
using HalResPtr = std::shared_ptr<HalResBase>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSR_RUNTIME_DEVICE_HAL_RES_BASE_H_
