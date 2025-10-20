/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_manager.h"

#include <algorithm>
#include <string>
#include <unordered_map>

#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
void AscendMemoryManager::Initialize() { (void)AscendMemAdapter::GetInstance()->Initialize(); }

void AscendMemoryManager::Finalize() {
  AscendMemoryPool::GetInstance().ReleaseDeviceRes();
  (void)AscendMemAdapter::GetInstance()->DeInitialize();
}

void AscendMemoryManager::ResetDynamicMemory() { AscendMemAdapter::GetInstance()->ResetDynamicMemory(); }

void AscendMemoryManager::ClearGlobalIdleMem() { AscendMemoryPool::GetInstance().ResetIdleMemBuf(); }

uint64_t AscendMemoryManager::GetMsMaxMemSize() const { return AscendMemAdapter::GetInstance()->MaxHbmSizeForMs(); }

uint64_t AscendMemoryManager::GetMsUsedHbmSize() const { return AscendMemAdapter::GetInstance()->GetMsUsedHbmSize(); }

void *AscendMemoryManager::MallocMemFromMemPool(size_t size, bool from_persistent_mem, bool need_recycle,
                                                uint32_t stream_id) {
  auto align_size = GetCommonAlignSize(size);
  return AscendMemoryPool::GetInstance().AllocTensorMem(align_size, from_persistent_mem, need_recycle, stream_id);
}

void AscendMemoryManager::FreeMemFromMemPool(void *device_ptr) {
  AscendMemoryPool::GetInstance().FreeTensorMem(device_ptr);
}

size_t AscendMemoryManager::GetMaxUsedMemorySize() const { return AscendMemoryPool::GetInstance().GetMaxUsedMemSize(); }

// Relevant function to manage memory statistics
size_t AscendMemoryManager::GetTotalMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalMemStatistics();
}
size_t AscendMemoryManager::GetTotalUsedMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
}
size_t AscendMemoryManager::GetTotalIdleMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalIdleMemStatistics();
}
size_t AscendMemoryManager::GetTotalEagerFreeMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalEagerFreeMemStatistics();
}
size_t AscendMemoryManager::GetUsedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemAllocatedStatistics();
}
size_t AscendMemoryManager::GetReservedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemReservedStatistics();
}
std::unordered_map<std::string, std::size_t> AscendMemoryManager::GetBlockCountsStatistics() const {
  return AscendMemoryPool::GetInstance().BlockCountsStatistics();
}
std::unordered_map<std::string, std::size_t> AscendMemoryManager::GetBlockUnitSizeStatistics() const {
  return AscendMemoryPool::GetInstance().BlockUnitSizeStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AscendMemoryManager::GetCommonMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().CommonMemBlocksInfoStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AscendMemoryManager::GetPersistentMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().PersistentMemBlocksInfoStatistics();
}
void AscendMemoryManager::ResetMaxMemoryReserved() { AscendMemoryPool::GetInstance().ResetMaxMemReserved(); }
void AscendMemoryManager::ResetMaxMemoryAllocated() { AscendMemoryPool::GetInstance().ResetMaxMemAllocated(); }
size_t AscendMemoryManager::EmptyCache() { return AscendMemoryPool::GetInstance().EmptyCache(); }

uint8_t *AscendMemoryManager::MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  MS_LOG(INFO) << "Malloc Memory for Static: size[" << align_size << "] communication_mem:" << communication_mem;

  uint8_t *alloc_address = reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(align_size));
  if (alloc_address != nullptr) {
    // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
    return communication_mem ? alloc_address + kMemAlignSize : alloc_address;
  }
  MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Fail to alloc memory, size: " << align_size
                    << "B, memory statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
}

uint8_t *AscendMemoryManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  MS_LOG(INFO) << "Malloc Memory for Dynamic: size[" << align_size << "] communication_mem: " << communication_mem;

  uint8_t *alloc_address =
    reinterpret_cast<uint8_t *>(AscendMemAdapter::GetInstance()->MallocDynamicDevMem(align_size));
  MS_EXCEPTION_IF_NULL(alloc_address);
  // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
  return communication_mem ? alloc_address + kMemAlignSize : alloc_address;
}

bool AscendMemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list, size_t /* total_size */,
                                                         std::vector<size_t> size_list, uint32_t stream_id) {
  auto device_ptr_list = MallocContinuousMemFromMemPool(size_list, stream_id);
  if (device_ptr_list.empty()) {
    return false;
  }
  if (addr_list.size() != device_ptr_list.size()) {
    MS_LOG(EXCEPTION) << "The size of device list " << addr_list.size() << " is not equal to the size of address list "
                      << device_ptr_list.size();
  }
  for (size_t i = 0; i < addr_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(device_ptr_list[i]);
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    addr_list[i]->SetDevicePtr(device_ptr_list[i]);
    addr_list[i]->set_from_mem_pool(true);
  }
  return true;
}

size_t AscendMemoryManager::GetAvailableMemSize() {
  auto available_mem_size = AscendMemoryPool::GetInstance().free_mem_size() +
                            AscendMemoryPool::GetInstance().TotalMemStatistics() -
                            AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
  return available_mem_size;
}

void AscendMemoryManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpy, device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpyAsync failed.";
    }
    if (CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1) != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "Call runtime aclrtSynchronizeStreamWithTimeout error.";
    }
  }
}

void AscendMemoryManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpy, host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpyAsync failed.";
    }
    if (CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1) != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "Call runtime aclrtSynchronizeStreamWithTimeout error.";
    }
  }
}

DynamicMemPool *AscendMemoryManager::GetMemoryPool() {
  if (MS_UNLIKELY(memory_pool_ == nullptr)) {
    memory_pool_ = &(AscendMemoryPool::GetInstance());
  }
  return memory_pool_;
}

void EnhancedAscendMemoryManager::Initialize() {
  AscendMemoryManager::Initialize();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK) << "EnhancedAscendMemoryManager initialize.";
  alloc_costs_.clear();
}

void EnhancedAscendMemoryManager::Finalize() {
  AscendMemoryManager::Finalize();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK) << "EnhancedAscendMemoryManager finalize";
  std::sort(alloc_costs_.begin(), alloc_costs_.end());
  // Calculate mean and median, then print them.
  auto total_size = alloc_costs_.size();
  if (total_size == 0) {
    MS_LOG(WARNING) << "No memory operation.";
    return;
  }
  double median = 0;
  if (total_size & 1) {
    median = (alloc_costs_[total_size >> 1] + alloc_costs_[(total_size >> 1) + 1]) >> 1;
  } else {
    median = alloc_costs_[total_size >> 1];
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK) << "EnhancedAscendMemoryManager median : " << median << "ns.";

  double sum = std::accumulate(alloc_costs_.begin(), alloc_costs_.end(), 0.0);
  double mean = sum / total_size;
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK) << "EnhancedAscendMemoryManager mean : " << mean << "ns.";

  const double cost_high_water = 1800;
  if (median > cost_high_water || mean > cost_high_water) {
    MS_LOG(WARNING) << "EnhancedAscendMemoryManager check failed, median : " << median << ", mean : " << mean;
  }
}

void *EnhancedAscendMemoryManager::MallocMemFromMemPool(size_t size, bool from_persistent_mem, bool need_recycle,
                                                        uint32_t stream_id) {
  auto start_tick = GetCurrentTick();
  auto ret = AscendMemoryManager::MallocMemFromMemPool(size, from_persistent_mem, need_recycle, stream_id);
  auto cost = GetCurrentTick() - start_tick;
  (void)alloc_costs_.emplace_back(cost);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK) << "Malloc memory cost : " << cost << "ns.";
  return ret;
}

bool EnhancedAscendMemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list,
                                                                 size_t total_size, std::vector<size_t> size_list,
                                                                 uint32_t stream_id) {
  auto start_tick = GetCurrentTick();
  auto ret = AscendMemoryManager::MallocContinuousMemFromMemPool(addr_list, total_size, size_list, stream_id);
  auto cost = GetCurrentTick() - start_tick;
  (void)alloc_costs_.emplace_back(cost);
  return ret;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
