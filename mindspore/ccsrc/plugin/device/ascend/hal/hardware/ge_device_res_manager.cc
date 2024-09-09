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

#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "hal/device/mbuf_receive_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include <utility>
#include <unordered_set>

#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_vmm_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#include "plugin/device/ascend/hal/device/ascend_pin_mem_pool.h"
#include "plugin/device/ascend/hal/special/parameter_replication.h"
#include "plugin/device/cpu/hal/device/cpu_device_synchronizer.h"
#include "mindspore/ops/kernel/ascend/pyboost/customize/stress_detect.h"
#include "include/transform/graph_ir/utils.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/acl_ir/op_api_util.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "utils/file_utils.h"
#include "graph/def_types.h"
#include "runtime/device/move_to.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace device {
namespace ascend {
using DeviceTensorStore = mindspore::runtime::DeviceTensorStore;
using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;

::ge::MemBlock *GeAllocator::Malloc(size_t size) {
  auto addr = res_manager_->AllocateMemory(size);
  MS_LOG(DEBUG) << "GE Allocator malloc addr: " << addr << " size: " << size;
  if (addr == nullptr) {
    MS_LOG(ERROR) << "GE Allocator malloc addr failed.";
    return nullptr;
  }
  auto mem_block = new ::ge::MemBlock(*this, addr, size);
  return mem_block;
}

void GeAllocator::Free(::ge::MemBlock *block) {
  res_manager_->FreeMemory(block->GetAddr());
  MS_LOG(DEBUG) << "GE Allocator free addr: " << block->GetAddr();
  delete block;
}

void GeDeviceResManager::Initialize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  if (!runtime_instance_->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  mem_manager_ = runtime_instance_->GetMemoryManager();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    swap_manager_ = std::make_shared<SwapManager>(kDefaultStreamIndex, &AscendMemoryPool::GetInstance(),
                                                  &AscendPinMemPool::GetInstance());
  }
}

void GeDeviceResManager::SetCPUMemManager() {
  if (is_use_cpu_memory_) {
    return;
  }
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
  runtime_instance_ = nullptr;
  mem_manager_ = std::make_shared<cpu::CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  is_use_cpu_memory_ = true;
}

void GeDeviceResManager::Destroy() {
  (void)DestroyAllEvents();
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

bool GeDeviceResManager::IsEnableVmm() const { return AscendVmmAdapter::GetInstance().IsEnabled(); }

bool GeDeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (IsEnableRefMode() && device_name_in_address != device_context_->device_context_key().device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_->device_context_key().device_name_;
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (runtime_instance_ != nullptr) {
    runtime_instance_->SetContext();
  }
  void *device_ptr = nullptr;

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  const auto kernel_tensor = address->kernel_tensor();
  const auto &hete_info = kernel_tensor == nullptr ? nullptr : kernel_tensor->heterogeneous_info();
  if (swap_manager_ != nullptr && hete_info != nullptr) {
    if (hete_info->need_alloc_hete_res_ == kernel::NeedAllocateHeteRes::NeedHostMem) {
      if (hete_info->host_ptr_ != nullptr) {
        MS_LOG(ERROR) << "Memory leak detected!";
        return false;
      }
      auto host_ptr = swap_manager_->AllocHostMemory(address->GetSize());
      hete_info->host_ptr_ = host_ptr;
      address->set_from_mem_pool(true);
      return true;
    }
    if (hete_info->need_alloc_hete_res_ == kernel::NeedAllocateHeteRes::NeedDiskFile) {
      if (!hete_info->file_name_.empty()) {
        MS_LOG(ERROR) << "Memory leak detected!";
        return false;
      }
      auto file_name = swap_manager_->GetSwapFileName(device_context_->device_context_key_.device_id_);
      swap_manager_->CreateFile(file_name, address->GetSize());
      hete_info->file_name_ = file_name;
      return true;
    }
  }
  if (swap_manager_ != nullptr) {
    device_ptr = swap_manager_->AllocDeviceMemory(address->GetSize(), stream_id);
  } else {
    device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(),
                                                    address->need_recycle(), stream_id);
  }

  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  return true;
}

void *GeDeviceResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void *GeDeviceResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return mem_manager_->MallocMemFromMemPool(size, true, false, stream_id);
}

size_t GeDeviceResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetMaxUsedMemorySize();
}

void GeDeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  const auto &hete_info =
    address->kernel_tensor() == nullptr ? nullptr : address->kernel_tensor()->heterogeneous_info();
  if (hete_info != nullptr && swap_manager_ != nullptr) {
    if (hete_info->host_ptr_ != nullptr) {
      swap_manager_->FreeHostMemory(hete_info->host_ptr_);
      hete_info->host_ptr_ = nullptr;
    }
    if (!hete_info->file_name_.empty()) {
      swap_manager_->DeleteFile(hete_info->file_name_);
      hete_info->file_name_ = "";
    }
  }
  if (address->GetPtr() != nullptr) {
    DeviceResManager::FreeMemory(address);
  }
}

void GeDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void GeDeviceResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                         const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void GeDeviceResManager::DefragMemory() { AscendMemoryPool::GetInstance().DefragMemory(); }

// Relevant function to manage memory statistics
size_t GeDeviceResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}

size_t GeDeviceResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}

size_t GeDeviceResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}

size_t GeDeviceResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}

size_t GeDeviceResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}

size_t GeDeviceResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> GeDeviceResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> GeDeviceResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}

DeviceMemInfo GeDeviceResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}

DeviceMemInfo GeDeviceResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}

void GeDeviceResManager::ResetMaxMemoryReserved() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->memory_pool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemReserved();
}

void GeDeviceResManager::ResetMaxMemoryAllocated() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->memory_pool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemAllocated();
}

void GeDeviceResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void GeDeviceResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> GeDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                                 uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceContinuousMem(aligned_size_list, stream_id);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(aligned_size_list, stream_id);
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (!is_use_cpu_memory_) {
    if (kernel_tensor->device_name().empty()) {
      kernel_tensor->set_device_name(device_context_->device_context_key().device_name_);
      kernel_tensor->set_device_id(device_context_->device_context_key().device_id_);
    }
    auto device_address = std::make_shared<AscendDeviceAddress>(kernel_tensor);
    device_address->set_device_synchronizer(std::make_shared<AscendDeviceSynchronizer>());
    return device_address;
  } else {
    if (kernel_tensor->device_name().empty()) {
      kernel_tensor->set_device_name(kCPUDevice);
      kernel_tensor->set_device_id(0);
    }
    auto device_address = std::make_shared<cpu::CPUDeviceAddress>(kernel_tensor);
    device_address->set_device_synchronizer(std::make_shared<cpu::CPUDeviceSynchronizer>());
    return device_address;
  }
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                         const Format &format, TypeId type_id,
                                                         const std::string &device_name, uint32_t device_id,
                                                         uint32_t stream_id) const {
  if (!is_use_cpu_memory_) {
    return std::make_shared<AscendDeviceAddress>(ptr, size, shape_vector, format, type_id, device_name, device_id,
                                                 stream_id);
  } else {
    return std::make_shared<cpu::CPUDeviceAddress>(ptr, size, shape_vector, format, type_id, kCPUDevice, device_id,
                                                   stream_id);
  }
}

void GeDeviceResManager::GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                             transform::SessionOptions *options) {
  MS_EXCEPTION_IF_NULL(options);
  if (ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.graphMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.variableMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

  auto atomic_clean_policy = ms_context_ptr->get_param<std::string>(MS_CTX_ATOMIC_CLEAN_POLICY);
  if (atomic_clean_policy.empty()) {
    atomic_clean_policy = "1";
  }
  (*options)["ge.exec.atomicCleanPolicy"] = atomic_clean_policy;
  MS_LOG(INFO) << "Set GE atomic clean policy to " << atomic_clean_policy << ".";
  (*options)["ge.graphRunMode"] = "1";
}

void GeDeviceResManager::CreateSessionAndGraphRunner() {
  std::shared_ptr<::ge::Session> sess = transform::GetGeSession();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (sess == nullptr) {
    transform::SessionOptions options;
    options["ge.enablePrintOpPass"] = "0";
    GeSetContextOptions(ms_context, &options);
    options["ge.constLifecycle"] = "graph";

    options["ge.exec.formatMode"] = "0";
    auto format_mode = common::GetEnv("MS_FORMAT_MODE");
    if (format_mode == "1" || (format_mode.empty() && ms_context->ascend_soc_version() != "ascend910")) {
      MS_LOG(INFO) << "Set GE option ge.exec.formatMode to 1.";
      options["ge.exec.formatMode"] = "1";
    }

    SetPassthroughGeOptions(false, &options);

    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = transform::NewGraphRunner(options);
  transform::SetGraphRunner(graph_runner);
}

bool GeDeviceResManager::LoadCollectiveCommLib() {
  // If this is simulation, load dummy collective communication library.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    collective_comm_lib_ = &DummyAscendCollectiveCommLib::GetInstance();
    return true;
  }
  // Load Multi ascend collective communication lib using dynamic library.
  collective_comm_lib_ = &MultiAscendCollectiveCommLib::GetInstance();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  MS_LOG(INFO) << "Loading MACCL collective library successfully.";
  return true;
}

bool GeDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  static thread_local std::once_flag is_set;
  std::call_once(is_set, []() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id));
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
    transform::AclUtil::SetDeterministic();
  });

  if (runtime_instance_ != nullptr) {
    if (force_bind) {
      runtime_instance_->SetContextForce();
    } else {
      runtime_instance_->SetContext();
    }
  }
  return true;
}

void GeDeviceResManager::ResetStreamAndCtx() {
  if (runtime_instance_ != nullptr) {
    runtime_instance_->ResetStreamAndCtx();
  }
}

bool GeDeviceResManager::CreateStream(size_t *stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool GeDeviceResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(stream_id, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       IntToUint(priority));
  return true;
}

size_t GeDeviceResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> GeDeviceResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

bool GeDeviceResManager::single_op_multi_stream_enable() const {
  return AscendStreamMng::GetInstance().single_op_multi_stream_enable();
}

void GeDeviceResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *GeDeviceResManager::GetStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(stream_id);
}

void GeDeviceResManager::SetCurrentStreamId(size_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(stream_id);
}

size_t GeDeviceResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool GeDeviceResManager::QueryStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(stream_id);
}

bool GeDeviceResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool GeDeviceResManager::SyncAllStreams() const {
  if (runtime_instance_ == nullptr) {
    return true;
  }
  runtime_instance_->SetContext();
  return AscendStreamMng::GetInstance().SyncAllStreams();
}

bool GeDeviceResManager::SyncNotDefaultStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncNotDefaultStreams();
}

size_t GeDeviceResManager::DefaultStream() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().default_stream_id();
}

std::pair<vector<size_t>, vector<size_t>> GeDeviceResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::unordered_set<tensor::TensorPtr> unique_list;
  vector<size_t> before_padding_sizes;
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    if (!unique_list.insert(tensor).second) {
      MS_LOG(EXCEPTION) << "Tensor input should be unique. Tensor[" << i << "], " << tensor->ToString();
    }
    auto real_size = tensor->Size();
    if (tensor->device_address() != nullptr) {
      const auto &device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
      real_size = device_address->GetSize();
    }
    before_padding_sizes.emplace_back(real_size);
  }
  std::vector<size_t> after_padding_sizes;
  for (auto &size : before_padding_sizes) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    after_padding_sizes.emplace_back(align_size);
  }
  auto stream_id = DefaultStream();
  auto device_ptr_list = AllocateContinuousMemory(before_padding_sizes, stream_id);
  for (size_t i = 0; i < after_padding_sizes.size(); ++i) {
    auto acl_ret = CALL_ASCEND_API(aclrtMemset, device_ptr_list[i], after_padding_sizes[i], 0, after_padding_sizes[i]);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(EXCEPTION) << "Clear overflow memory failed, aclrtMemset size = " << after_padding_sizes[i]
                        << ", ret = " << acl_ret;
    }
    MS_LOG(DEBUG) << "Clear ptr:" << device_ptr_list[i] << ", size:" << after_padding_sizes[i];
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // create device for all tensor in tensor list
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    const auto &ptr = device_ptr_list[i];
    auto device_address = CreateDeviceAddress(ptr, before_padding_sizes[i], tensor->shape(), Format::ND,
                                              tensor->data_type(), device_name, device_id, stream_id);
    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << ptr << ", size:" << before_padding_sizes[i]
                  << ", shape:" << tensor->shape() << ", data_type:" << TypeIdToString(tensor->data_type());
    MS_EXCEPTION_IF_NULL(device_address);
    if (tensor->device_address() == nullptr) {
      device_address->SyncHostToDevice(before_padding_sizes[i], tensor->data_c());
    } else {
      device_address->SyncDeviceToDevice(tensor->device_address().get());
    }
    tensor->set_device_address(device_address);
  }
  return std::make_pair(before_padding_sizes, after_padding_sizes);
}

TensorPtr GeDeviceResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                              const std::vector<size_t> &before_padding_size,
                                                              const std::vector<size_t> &after_padding_size,
                                                              size_t start, size_t end) {
  if (start >= tensor_list.size() || end > tensor_list.size()) {
    MS_EXCEPTION(ValueError) << "start:" << start << ", end:" << end << ", but tensor_list size:" << tensor_list.size();
  }
  size_t size = std::accumulate(after_padding_size.begin() + start, after_padding_size.begin() + end - 1,
                                before_padding_size[end - 1]);
  ShapeVector shape = {int64_t(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = std::make_shared<Tensor>(tensor_list[start]->data_type(), shape);
  MS_EXCEPTION_IF_NULL(tensor_list[start]->device_address());
  auto ptr = tensor_list[start]->device_address()->GetMutablePtr();

  auto stream_id = DefaultStream();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address =
    CreateDeviceAddress(ptr, size, shape, Format::ND, tensor->data_type(), device_name, device_id, stream_id);
  tensor->set_device_address(device_address);
  return tensor;
}

TensorPtr GeDeviceResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                           size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = std::make_shared<Tensor>(type_id, shape);
  MS_EXCEPTION_IF_NULL(first_tensor->device_address());
  auto ptr = first_tensor->device_address()->GetMutablePtr();
  auto offset_size = start * type_size;

  auto stream_id = DefaultStream();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address = CreateDeviceAddress(reinterpret_cast<uint8_t *>(ptr) + offset_size, tensor_size, shape,
                                            Format::ND, type_id, device_name, device_id, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
  tensor->set_device_address(device_address);
  return tensor;
}

int GeDeviceResManager::StressDetect() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return 0;
  }
  MS_EXCEPTION_IF_NULL(device_context_);
  return kernel::pyboost::StressDetectKernel(device_context_);
}

// return 0 when success, otherwise return 1
int GeDeviceResManager::SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank) const {
  ParamReplication replicator(this);
  replicator.Init();
  return replicator.SendRecv(params, src_rank, dst_rank);
}

int GeDeviceResManager::CleanTdtChannel() const {
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
DeviceEventPtr GeDeviceResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }

  uint32_t flag = 0;
  if (enable_blocking) {
    flag |= ACL_EVENT_SYNC;
  }
  if (enable_record_wait) {
    flag |= ACL_EVENT_CAPTURE_STREAM_PROGRESS;
  }
  return std::make_shared<AscendEvent>(flag);
}

DeviceEventPtr GeDeviceResManager::CreateEventWithFlag(bool enable_timing, bool blocking) {
  auto flag = enable_timing ? ACL_EVENT_TIME_LINE : ACL_EVENT_DEFAULT;
  auto event = std::make_shared<AscendEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

void GeDeviceResManager::MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor,
                                const std::string &to, bool blocking, bool *return_self) {
  device::MoveTo(src_tensor, dst_tensor, to, blocking, return_self);
}

std::vector<device::DeviceMemPtr> GeDeviceResManager::GetMemUceInfo(int32_t device_id) {
  aclrtMemUceInfo info[MAX_MEM_UCE_INFO_ARRAY_SIZE];
  size_t retSize = 0;
  auto ret = CALL_ASCEND_API(aclrtGetMemUceInfo, device_id, info, sizeof(info) / sizeof(aclrtMemUceInfo), &retSize);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtGetMemUceInfo failed.";
  }
  if (retSize == 0) {
    MS_LOG(EXCEPTION) << "aclrtGetMemUceInfo get UCE Error failed.";
  }

  MS_LOG(INFO) << "aclrtGetMemUceInfo get UCE Error, retSize is " << retSize;

  MemUceInfo mem_uce_info;
  mem_uce_info.device_id = device_id;
  mem_uce_info.info.assign(info, info + retSize);
  mem_uce_info.retSize = retSize;

  std::lock_guard<std::mutex> lock(mem_uce_info_mutex_);
  mem_uce_info_ = mem_uce_info;

  std::vector<device::DeviceMemPtr> mem_uce_device_list;
  for (size_t i = 0; i < retSize; ++i) {
    mem_uce_device_list.emplace_back(info[i].addr);
  }
  return mem_uce_device_list;
}

bool GetUceLevelWithMemPoolForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                  const DeviceMemInfo &common_mem_blocks_info, const MemUceInfo &mem_uce_info) {
  for (auto iter = persistent_mem_blocks_info.begin(); iter != persistent_mem_blocks_info.end(); ++iter) {
    auto persistent_block_start_addr = reinterpret_cast<char *>(iter->first);
    auto block_info = iter->second.begin();
    auto persistent_block_end_addr = persistent_block_start_addr + block_info->second;
    for (size_t i = 0; i < mem_uce_info.info.size(); ++i) {
      auto mem_uce_start_addr = reinterpret_cast<char *>(mem_uce_info.info[i].addr);
      auto mem_uce_end_addr = mem_uce_start_addr + mem_uce_info.info[i].len;
      if ((persistent_block_end_addr >= mem_uce_start_addr && persistent_block_start_addr < mem_uce_start_addr) ||
          (mem_uce_end_addr >= persistent_block_start_addr && mem_uce_start_addr < persistent_block_start_addr)) {
        MS_LOG(DEBUG) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }

  for (auto iter = common_mem_blocks_info.begin(); iter != common_mem_blocks_info.end(); ++iter) {
    auto common_block_start_addr = reinterpret_cast<char *>(iter->first);
    auto block_info = iter->second.begin();
    auto common_block_end_addr = common_block_start_addr + block_info->second;
    for (size_t i = 0; i < mem_uce_info.info.size(); ++i) {
      auto mem_uce_start_addr = reinterpret_cast<char *>(mem_uce_info.info[i].addr);
      auto mem_uce_end_addr = mem_uce_start_addr + mem_uce_info.info[i].len;
      if ((common_block_end_addr >= mem_uce_start_addr && common_block_start_addr < mem_uce_start_addr) ||
          (mem_uce_end_addr >= common_block_start_addr && mem_uce_start_addr < common_block_start_addr)) {
        MS_LOG(DEBUG) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }
  return false;
}

std::string GetUceProcessStrategyForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                        const DeviceMemInfo &common_mem_blocks_info, const MemUceInfo &mem_uce_info) {
  // Judge whether weights got uce error.
  MS_LOG(INFO) << "Start to get UCE process strategy for kbk.";
  const auto &device_tensors = DeviceTensorStore::GetInstance().GetAll();
  for (auto iter = device_tensors.begin(); iter != device_tensors.end(); ++iter) {
    auto device_tensor_list = iter->second;
    for (const auto &device_tensor : device_tensor_list) {
      MS_EXCEPTION_IF_NULL(device_tensor);
      auto device_tensor_start_addr = reinterpret_cast<char *>(const_cast<void *>(device_tensor->GetPtr()));
      auto device_tensor_end_addr = device_tensor_start_addr + device_tensor->GetSize();
      for (size_t i = 0; i < mem_uce_info.info.size(); ++i) {
        auto mem_uce_start_addr = reinterpret_cast<char *>(mem_uce_info.info[i].addr);
        auto mem_uce_end_addr = mem_uce_start_addr + mem_uce_info.info[i].len;
        // Return RS_UCE_HIGHLEVEL if overlap of device tensor addr and mem uce addr.
        if ((device_tensor_end_addr >= mem_uce_start_addr && device_tensor_start_addr < mem_uce_start_addr) ||
            (mem_uce_end_addr >= device_tensor_start_addr && mem_uce_start_addr < device_tensor_start_addr)) {
          MS_LOG(DEBUG) << "UCE process strategy is RS_UCE_HIGHLEVEL.";
          return RS_UCE_HIGHLEVEL;
        }
      }
    }
  }

  // Return RS_UCE_LOWLEVEL if overlap of memory pool addr and mem uce addr.
  if (GetUceLevelWithMemPoolForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_info)) {
    return RS_UCE_LOWLEVEL;
  }

  MS_LOG(DEBUG) << "UCE process strategy is RS_NORMAL.";

  return RS_NORMAL;
}

std::string GeDeviceResManager::GetUceProcessStrategy() const {
  auto persistent_mem_blocks_info = GetPersistentMemBlocksInfoStatistics();
  auto common_mem_blocks_info = GetCommonMemBlocksInfoStatistics();
  return GetUceProcessStrategyForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_info_);
}

void GeDeviceResManager::UceMemRepair(int32_t device_id) {
  if (device_id != mem_uce_info_.device_id) {
    MS_LOG(EXCEPTION) << "Uce mem repair device id is not correct, device id is " << mem_uce_info_.device_id
                      << ", but got " << device_id << ".";
  }
  aclrtMemUceInfo *info = mem_uce_info_.info.data();
  if (CALL_ASCEND_API(aclrtMemUceRepair, mem_uce_info_.device_id, info, mem_uce_info_.retSize) != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtMemUceRepair failed.";
  }
  // Clear mem_uce_info.
  mem_uce_info_.device_id = 0;
  mem_uce_info_.info.clear();
  mem_uce_info_.retSize = 0;
}

void GeDeviceResManager::StopDevice(int32_t device_id) {
  UCEException::GetInstance().set_force_stop_flag(true);

  uint32_t timeout = 0;
  if (CALL_ASCEND_API(aclrtDeviceTaskAbort, device_id, timeout) != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtDeviceTaskAbort failed.";
  }
}

void GeDeviceResManager::ThrowUCEError() {
  UCEException::GetInstance().set_uce_flag(true);
  MS_EXCEPTION(UCEError) << "UCEError.";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
