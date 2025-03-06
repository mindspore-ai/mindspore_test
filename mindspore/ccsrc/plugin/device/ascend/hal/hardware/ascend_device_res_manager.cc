/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "plugin/device/ascend/hal/device/tensorreport_utils.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>

#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "plugin/device/ascend/hal/special/parameter_replication.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/device/ascend/hal/hardware/stress_detect.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/move_to.h"
#include "utils/ms_exception.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
void AscendDeviceResManager::Initialize() {
  if (initialized_) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  if (!runtime_instance_->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  ResKey res_key{DeviceType::kAscend, device_id};
  ascend_res_manager_ = dynamic_cast<AscendResManager *>(HalResManager::GetInstance().GetOrCreateResManager(res_key));
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  ascend_res_manager_->Initialize();
  initialized_ = true;
}

void AscendDeviceResManager::Destroy() {
  if (!initialized_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  (void)ascend_res_manager_->DestroyAllEvents();
  // release runtime
  if (runtime_instance_ != nullptr) {
    runtime_instance_->ReleaseDeviceRes();
    runtime_instance_ = nullptr;
  }
  ascend_res_manager_->Destroy();
  initialized_ = false;
}

void AscendDeviceResManager::SetDeterministic() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SetDeterministic();
}

bool AscendDeviceResManager::IsEnableVmm() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->IsEnableVmm();
}
void AscendDeviceResManager::SetAclDeterministic() { return ascend_res_manager_->SetAclDeterministic(); }

bool AscendDeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocateMemory(address, stream_id);
}

bool AscendDeviceResManager::AllocateForHete(mindspore::device::DeviceAddress *const &address,
                                             mindspore::kernel::HeterogeneousInfoPtr hete_info) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocateForHete(address, hete_info);
}

void *AscendDeviceResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocateMemory(size, stream_id);
}

void *AscendDeviceResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocateStaticMemory(size, stream_id);
}

size_t AscendDeviceResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetMaxUsedMemorySize();
}

void AscendDeviceResManager::FreeForHete(mindspore::kernel::HeterogeneousInfoPtr hete_info) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->FreeForHete(hete_info);
}

void AscendDeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->FreeMemory(address);
}

void AscendDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->FreeMemory(ptr);
}

void AscendDeviceResManager::FreePartMemorys(const std::vector<void *> &free_addrs,
                                             const std::vector<void *> &keep_addrs,
                                             const std::vector<size_t> &keep_addr_sizes) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->FreePartMemorys(free_addrs, keep_addrs, keep_addr_sizes);
}

void AscendDeviceResManager::DefragMemory() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->DefragMemory();
}

// Relevant function to manage memory statistics
size_t AscendDeviceResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetTotalMemStatistics();
}

size_t AscendDeviceResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetTotalUsedMemStatistics();
}

size_t AscendDeviceResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetTotalIdleMemStatistics();
}

size_t AscendDeviceResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetTotalEagerFreeMemStatistics();
}

size_t AscendDeviceResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetUsedMemPeakStatistics();
}

size_t AscendDeviceResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> AscendDeviceResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> AscendDeviceResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetBlockUnitSizeStatistics();
}

DeviceMemInfo AscendDeviceResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetCommonMemBlocksInfoStatistics();
}

DeviceMemInfo AscendDeviceResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetPersistentMemBlocksInfoStatistics();
}

void AscendDeviceResManager::ResetMaxMemoryReserved() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->ResetMaxMemoryReserved();
}

void AscendDeviceResManager::ResetMaxMemoryAllocated() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->ResetMaxMemoryAllocated();
}

void AscendDeviceResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void AscendDeviceResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> AscendDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                                     uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocateContinuousMemory(size_list, stream_id);
}

DeviceAddressPtr AscendDeviceResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateDeviceAddress(kernel_tensor);
}

DeviceAddressPtr AscendDeviceResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                             const Format &format, TypeId type_id,
                                                             const std::string &device_name, uint32_t device_id,
                                                             uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateDeviceAddress(ptr, size, shape_vector, format, type_id, device_name, device_id,
                                                  stream_id);
}

bool AscendDeviceResManager::LoadCollectiveCommLib() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->LoadCollectiveCommLib();
}

CollectiveCommunicationLib *AscendDeviceResManager::collective_comm_lib() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->collective_comm_lib();
}

std::shared_ptr<MemoryManager> AscendDeviceResManager::mem_manager() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->mem_manager();
}

std::shared_ptr<SwapManager> AscendDeviceResManager::swap_manager() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->swap_manager();
}

bool AscendDeviceResManager::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->DestroyEvent(event);
}

bool AscendDeviceResManager::DestroyAllEvents() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->DestroyAllEvents();
}

bool AscendDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->BindDeviceToCurrentThread(force_bind);
}

void AscendDeviceResManager::ResetStreamAndCtx() {
  if (runtime_instance_ != nullptr) {
    runtime_instance_->ResetStreamAndCtx();
  }
}

bool AscendDeviceResManager::CreateStream(size_t *stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateStream(stream_id);
}

bool AscendDeviceResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateStreamWithPriority(stream_id, priority);
}

size_t AscendDeviceResManager::QueryStreamSize() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->QueryStreamSize();
}

std::vector<uint32_t> AscendDeviceResManager::GetStreamIds() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetStreamIds();
}

bool AscendDeviceResManager::single_op_multi_stream_enable() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->single_op_multi_stream_enable();
}

void AscendDeviceResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *AscendDeviceResManager::GetStream(size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetStream(stream_id);
}

size_t AscendDeviceResManager::GetCommunicationStreamID() const {
  if (runtime_instance_ == nullptr) {
    MS_LOG(WARNING) << "runtime_instance_ is nullptr, can not to get communication stream";
    return kDefaultStreamIndex;
  }
  return runtime_instance_->communication_stream_id();
}

size_t AscendDeviceResManager::GetCommunicationStreamIDByGroup(const std::string &group) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(EXCEPTION) << "Bind context to current thread failed";
    return 0;
  }
  if (runtime_instance_ == nullptr) {
    MS_LOG(WARNING) << "runtime_instance_ is nullptr, can not to get communication stream by group";
    return GetCommunicationStreamID();
  }
  return runtime_instance_->GetCommunicationStreamIDByGroup(group);
}

void AscendDeviceResManager::SetCurrentStreamId(size_t stream_id) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SetCurrentStreamId(stream_id);
}

size_t AscendDeviceResManager::GetCurrentStreamId() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetCurrentStreamId();
}

bool AscendDeviceResManager::QueryStream(size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->QueryStream(stream_id);
}

bool AscendDeviceResManager::SyncStream(size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SyncStream(stream_id);
}

bool AscendDeviceResManager::SyncAllStreams() const {
  // AscendDeviceResManager is not be initialized and ascend_res_manager_ is nullptr when dryrun or export graph.
  if (ascend_res_manager_ == nullptr) {
    MS_LOG(WARNING) << "The ascend_res_manager_ is nullptr in scenarios where it is not actually executed";
    return true;
  }
  return ascend_res_manager_->SyncAllStreams();
}

bool AscendDeviceResManager::SyncNotDefaultStreams() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->SyncNotDefaultStreams();
}

size_t AscendDeviceResManager::DefaultStream() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->DefaultStream();
}

std::pair<std::vector<size_t>, std::vector<size_t>> AscendDeviceResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->AllocDeviceMemoryForTensorList(tensor_list, enable_mem_align);
}

TensorPtr AscendDeviceResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                                  const std::vector<size_t> &before_padding_size,
                                                                  const std::vector<size_t> &after_padding_size,
                                                                  size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetSliceByTensorListIndexHandle(tensor_list, before_padding_size, after_padding_size,
                                                              start, end);
}

TensorPtr AscendDeviceResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                               size_t end) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetSliceByPaddingShapeHandle(first_tensor, start, end);
}

int AscendDeviceResManager::StressDetect() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return 0;
  }
  MS_EXCEPTION_IF_NULL(device_context_);
  return kernel::pyboost::StressDetectKernel(device_context_);
}

// return 0 when success, otherwise return 1
int AscendDeviceResManager::SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank) const {
  ParamReplication replicator(this);
  replicator.Init();
  return replicator.SendRecv(params, src_rank, dst_rank);
}

int AscendDeviceResManager::ResetParams(const std::vector<tensor::TensorPtr> &params) const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->ResetParams(params);
}

int AscendDeviceResManager::CleanTdtChannel() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return 1;
  }
  MS_EXCEPTION_IF_NULL(device_context_);
  MbufDataHandlerManager::GetInstance().CleanChannels();
  (void)device::DataQueueMgr::GetInstance().CleanTdtHandle();
  return 0;
}

// ACL_EVENT_TIME_LINE: indicates that the number of created events is not limited, and the created events can be used
//  to compute the elapsed time between events, which may cause lost some performance.
// ACL_EVENT_SYNC: indicates that the number of created events is limited, and the created events can be used for
//  synchronization between multiple streams.
// ACL_EVENT_CAPTURE_STREAM_PROGRESS: indicates that the number of created events is not limited and high performance,
//  and the created events can not be used for timing and synchronization.
DeviceEventPtr AscendDeviceResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateRuntimeEvent(enable_blocking, enable_record_wait);
}

DeviceEventPtr AscendDeviceResManager::CreateEventWithFlag(bool enable_timing, bool blocking,
                                                           bool use_extensional_api) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->CreateEventWithFlag(enable_timing, blocking, use_extensional_api);
}

bool AscendDeviceResManager::GetMemUceInfo(int32_t device_id) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetMemUceInfo(device_id);
}

std::vector<uint64_t> AscendDeviceResManager::GetOptimizerTimestamps() {
  OptimizerEventInfo::GetInstance().GetOptimizerTimestamp(false);
  auto hbm_error_time = UCEException::GetInstance().get_uce_occur_time();
  auto opt_start_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_start_timestamp();
  auto opt_end_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_end_timestamp();
  return std::vector<uint64_t>{hbm_error_time, opt_start_timestamp, opt_end_timestamp};
}

std::vector<std::pair<device::DeviceMemPtr, size_t>> AscendDeviceResManager::GetMemUceAddr() {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetMemUceAddr();
}

void AscendDeviceResManager::UceMemRepair(int32_t device_id) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->UceMemRepair(device_id);
}

void AscendDeviceResManager::StopDevice(int32_t device_id) {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->StopDevice(device_id);
}

void *AscendDeviceResManager::GetCopyDataStream() const {
  MS_EXCEPTION_IF_NULL(ascend_res_manager_);
  return ascend_res_manager_->GetCopyDataStream();
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
