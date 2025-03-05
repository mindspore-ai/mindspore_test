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

#include "plugin/res_manager/ascend/ascend_res_manager.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <utility>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>

#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "plugin/res_manager/cpu/cpu_mem_manager/cpu_memory_manager.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_synchronizer.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_memory_manager.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_vmm_adapter.h"
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_address.h"
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_synchronizer.h"
#include "plugin/res_manager/ascend/event/ascend_event.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_pin_mem_pool.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_compiler_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "utils/file_utils.h"
#include "utils/distributed_meta.h"
#include "graph/def_types.h"
#include "runtime/device/move_to.h"
#include "runtime/device/tensor_array.h"
#include "acl/acl_rt.h"
#include "plugin/res_manager/ascend/collective/ccool_collective_comm_lib.h"
#include "plugin/res_manager/ascend/collective/multi_ascend_collective_comm_lib.h"
#include "plugin/res_manager/ascend/collective/ascend_collective_comm_lib.h"
#include "plugin/res_manager/ascend/collective/dummy_ascend_collective_comm_lib.h"
#ifdef ENABLE_INTERNAL_KERNELS
#include "plugin/res_manager/ascend/collective/lowlatency_collective_comm_lib.h"
#endif
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
typedef HcclResult (*HcclSetConfigFunc)(HcclConfig, HcclConfigValue);
std::mutex set_opt_mutex;
std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kLatest = "latest";
  auto pos = path_tmp.find(kLatest);
  if (pos == std::string::npos) {
    MS_EXCEPTION(ValueError)
      << "Get ascend path failed, please check whether CANN packages are installed correctly, \n"
         "and environment variables are set by source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh.";
  }
  return path_tmp.substr(0, pos) + kLatest + "/";
}

void *GetAclFunc(const std::string &lib_path, const std::string &func_name) {
  static auto ascend_path = GetAscendPath();
  auto load_path = ascend_path + "/lib64/" + lib_path;

  auto handler = dlopen(load_path.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << load_path << " failed!" << dlerror();
    return nullptr;
  }

  auto func = dlsym(handler, func_name.c_str());
  if (func == nullptr) {
    MS_LOG(INFO) << "Dlsym " << func_name << " from " << load_path << " failed!" << dlerror();
  }
  return func;
}

Format GetFormat(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto format = Format::DEFAULT_FORMAT;
  if (tensor->device_address() != nullptr) {
    const auto temp_device_address = tensor->device_address();
    auto const device_address = std::dynamic_pointer_cast<const DeviceAddress>(temp_device_address);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->device_name() != "CPU") {
      auto const src_device_address = std::dynamic_pointer_cast<const AscendDeviceAddress>(temp_device_address);
      MS_EXCEPTION_IF_NULL(src_device_address);
      format = FromStrToEnum(src_device_address->format());
    } else {
      tensor->data_sync();
      tensor->set_device_address(nullptr);
    }
  }
  return format;
}
}  // namespace

void AscendResManager::Initialize() {
  if (initialized_) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (!(IS_VLOG_ON(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK))) {
    mem_manager_ = std::make_shared<AscendMemoryManager>();
  } else {
    mem_manager_ = std::make_shared<EnhancedAscendMemoryManager>();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();
  swap_manager_ = std::make_shared<SwapManager>(kDefaultStreamIndex, &AscendMemoryPool::GetInstance(),
                                                &AscendPinMemPool::GetInstance());
  initialized_ = true;
}

void AscendResManager::Destroy() {
  if (!initialized_) {
    return;
  }
  (void)DestroyAllEvents();
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }

  initialized_ = false;
}

bool AscendResManager::IsEnableVmm() const { return AscendVmmAdapter::GetInstance().IsEnabled(); }

bool AscendResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  if (address->pointer_ref_count()->ptr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  void *device_ptr = nullptr;

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  const auto &hete_info =
    address->kernel_tensor() == nullptr ? nullptr : address->kernel_tensor()->heterogeneous_info();
  if (hete_info != nullptr) {
    return AllocateForHete(address, hete_info);
  }
  device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(),
                                                  address->need_recycle(), stream_id);
  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  static bool enable_memory_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (enable_memory_tracker) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  }
  return true;
}

bool AscendResManager::AllocateForHete(mindspore::device::DeviceAddress *const &address,
                                       mindspore::kernel::HeterogeneousInfoPtr hete_info) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(hete_info);
  if (hete_info->need_alloc_hete_res_ == kernel::NeedAllocateHeteRes::NeedHostMem) {
    if (hete_info->host_ptr_ != nullptr) {
      MS_LOG(ERROR) << "Memory leak detected!";
      return false;
    }
    auto host_ptr = swap_manager_->AllocHostMemory(address->GetSize());
    hete_info->host_ptr_ = host_ptr;
    address->set_from_mem_pool(true);
  }
  if (hete_info->need_alloc_hete_res_ == kernel::NeedAllocateHeteRes::NeedDiskFile) {
    if (!hete_info->file_name_.empty()) {
      MS_LOG(ERROR) << "Memory leak detected!";
      return false;
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto file_name = swap_manager_->GetSwapFileName(device_id);
    swap_manager_->CreateFile(file_name, address->GetSize());
    hete_info->file_name_ = file_name;
  }
  return true;
}
void *AscendResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void *AscendResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  return mem_manager_->MallocMemFromMemPool(size, true, false, stream_id);
}

size_t AscendResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetMaxUsedMemorySize();
}

void AscendResManager::FreeForHete(mindspore::kernel::HeterogeneousInfoPtr hete_info) const {
  MS_EXCEPTION_IF_NULL(hete_info);
  if (hete_info->host_ptr_ != nullptr) {
    swap_manager_->FreeHostMemory(hete_info->host_ptr_);
    hete_info->host_ptr_ = nullptr;
  }
  if (!hete_info->file_name_.empty()) {
    swap_manager_->DeleteFile(hete_info->file_name_);
    hete_info->file_name_ = "";
  }
}

void AscendResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  const auto &hete_info =
    address->kernel_tensor() == nullptr ? nullptr : address->kernel_tensor()->heterogeneous_info();
  if (hete_info != nullptr) {
    FreeForHete(hete_info);
  }

  void *device_ptr = address->GetMutablePtr();
  if (device_ptr != nullptr) {
    if (!address->from_mem_pool()) {
      MS_LOG(DEBUG) << "device address:" << address << " ptr:" << device_ptr << " not from pool";
      return;
    }

    MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << device_ptr;
    FreeMemory(device_ptr);
    address->set_ptr(nullptr);
  }
}

void AscendResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void AscendResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                       const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void AscendResManager::DefragMemory() { AscendMemoryPool::GetInstance().DefragMemory(); }

// Relevant function to manage memory statistics
size_t AscendResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}

size_t AscendResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}

size_t AscendResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}

size_t AscendResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}

size_t AscendResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}

size_t AscendResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}

DeviceMemInfo AscendResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}

DeviceMemInfo AscendResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}

void AscendResManager::ResetMaxMemoryReserved() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemReserved();
}

void AscendResManager::ResetMaxMemoryAllocated() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemAllocated();
}

void AscendResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void AscendResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> AscendResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                               uint32_t stream_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(aligned_size_list, stream_id);
}

DeviceAddressPtr AscendResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (!is_use_cpu_memory_) {
    if (kernel_tensor->device_name().empty()) {
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      auto device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      kernel_tensor->set_device_name(device_type);
      kernel_tensor->set_device_id(device_id);
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

DeviceAddressPtr AscendResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
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

bool AscendResManager::LoadCollectiveCommLib() {
  // If this is simulation, load dummy collective communication library.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    collective_comm_lib_ = &DummyAscendCollectiveCommLib::GetInstance();
    return true;
  }
  if (DistributedMeta::GetInstance()->enable_cross_cluster()) {
    collective_comm_lib_ = &CcoolCollectiveCommLib::GetInstance();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_);
    MS_LOG(INFO) << "Loading CCOOL collective library successfully.";
    return true;
  }
  // Load Multi ascend collective communication lib using dynamic library.
  collective_comm_lib_ = &MultiAscendCollectiveCommLib::GetInstance();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  MS_LOG(INFO) << "Loading MACCL collective library successfully.";
  return true;
}

void AscendResManager::SetDeterministic() const {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  if (UseSimulationApi()) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? true : false;
  MS_LOG(INFO) << "Set kernel deterministic value: " << (is_deterministic ? "1" : "0");
  // Set acl
  auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_DETERMINISTIC, is_deterministic ? "1" : "0");
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
  // Set acl sys
  ret = CALL_ASCEND_API(aclrtCtxSetSysParamOpt, aclSysParamOpt::ACL_OPT_DETERMINISTIC, is_deterministic ? 1 : 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl sys set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
  // Set hccl
  const std::string hccl_lib = "libhccl.so";
  const std::string hccl_set_config_name = "HcclSetConfig";
  auto hccl_set_config = GetAclFunc(hccl_lib, hccl_set_config_name);
  if (hccl_set_config == nullptr) {
    MS_LOG(EXCEPTION) << "Get 'HcclSetConfig' from " << hccl_lib << " failed!";
  }
  auto hccl_set_config_func = reinterpret_cast<HcclSetConfigFunc>(hccl_set_config);
  HcclConfigValue config = {is_deterministic ? 1 : 0};
  auto hccl_ret = hccl_set_config_func(HcclConfig::HCCL_DETERMINISTIC, config);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Hccl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
}

bool AscendResManager::BindDeviceToCurrentThread(bool force_bind) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  static thread_local std::once_flag is_set;
  std::call_once(is_set, [this, device_id]() {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id));
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
    SetDeterministic();
  });

  if (force_bind) {
    AscendHalManager::GetInstance().SetContextForce(device_id);
  } else {
    AscendHalManager::GetInstance().SetContext(device_id);
  }

  return true;
}

bool AscendResManager::CreateStream(size_t *stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool AscendResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(stream_id, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       IntToUint(priority));
  return true;
}

size_t AscendResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> AscendResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

bool AscendResManager::single_op_multi_stream_enable() const {
  return AscendStreamMng::GetInstance().single_op_multi_stream_enable();
}

void AscendResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *AscendResManager::GetStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(stream_id);
}

void AscendResManager::SetCurrentStreamId(size_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(stream_id);
}

size_t AscendResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool AscendResManager::QueryStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(stream_id);
}

bool AscendResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool AscendResManager::SyncAllStreams() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  return AscendStreamMng::GetInstance().SyncAllStreams();
}

bool AscendResManager::SyncNotDefaultStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncNotDefaultStreams();
}

size_t AscendResManager::DefaultStream() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().default_stream_id();
}

std::pair<std::vector<size_t>, std::vector<size_t>> AscendResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_LOG(INFO) << "Start AllocDeviceMemoryForTensorList";
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  if (enable_mem_align == false) {
    size_t total_size = std::accumulate(before_padding_sizes.begin(), before_padding_sizes.end(), IntToSize(0));
    auto stream_id = DefaultStream();
    auto total_align_size = device::MemoryManager::GetCommonAlignSize(total_size);
    auto device_ptr = mem_manager_->MallocMemFromMemPool(total_align_size, false, false, stream_id);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "PyNative", total_align_size, device_ptr,
                                                   memory::mem_pool::MemType::kContinuousMemory);
    if (!device_ptr) {
      MS_LOG(EXCEPTION) << "Alloc device memory failed!";
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

    // create device for all tensor in tensor list
    char *ptr = reinterpret_cast<char *>(device_ptr);
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      const auto &tensor = tensor_list[i];
      auto format = GetFormat(tensor);
      auto device_address = CreateDeviceAddress(reinterpret_cast<void *>(ptr), before_padding_sizes[i], tensor->shape(),
                                                format, tensor->data_type(), device_name, device_id, stream_id);
      MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(ptr)
                    << ", size:" << before_padding_sizes[i] << ", shape:" << tensor->shape()
                    << ", data_type:" << TypeIdToString(tensor->data_type());
      MS_EXCEPTION_IF_NULL(device_address);
      if (tensor->device_address() == nullptr) {
        device_address->SyncHostToDevice(before_padding_sizes[i], tensor->data_c());
      } else {
        device_address->SyncDeviceToDevice(tensor->device_address().get());
      }
      tensor->set_device_address(device_address);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PyNative", device_address->device_name(),
                                                     device_ptr, tensor->data_type(), tensor->shape(),
                                                     tensor->storage_info());
      ptr += before_padding_sizes[i];
    }
    std::vector<size_t> after_padding_sizes(before_padding_sizes.size());
    std::copy(before_padding_sizes.begin(), before_padding_sizes.end(), after_padding_sizes.begin());
    after_padding_sizes.back() = total_align_size - total_size + before_padding_sizes.back();
    return std::make_pair(before_padding_sizes, after_padding_sizes);
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
    auto format = GetFormat(tensor);
    auto device_address = CreateDeviceAddress(ptr, before_padding_sizes[i], tensor->shape(), format,
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
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "PyNative", before_padding_sizes[i], ptr,
                                                   memory::mem_pool::MemType::kContinuousMemory);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PyNative", device_address->device_name(), ptr,
                                                   tensor->data_type(), tensor->shape(), tensor->storage_info());
  }
  return std::make_pair(before_padding_sizes, after_padding_sizes);
}

int AscendResManager::ResetParams(const std::vector<tensor::TensorPtr> &params) const {
  constexpr size_t kDefaultStreamId = 0;
  auto stream_id = kDefaultStreamId;
  auto stream_ptr = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(INFO) << "Size of params is " << params.size();

  for (size_t index = 0; index < params.size(); ++index) {
    auto &tensor = params[index];
    if (tensor->device_address() == nullptr || tensor->device_address()->GetMutablePtr() == nullptr) {
      MS_LOG(INFO) << "Parameter " << index << "/" << params.size() << " size=" << tensor->Size()
                   << " tensor device address is nullptr, skip resetting.";
      continue;
    }
    MS_LOG(INFO) << "Parameter " << index << "/" << params.size() << " size=" << tensor->Size();
    auto ret = CALL_ASCEND_API(aclrtMemsetAsync, tensor->device_address()->GetMutablePtr(), tensor->Size(), 0,
                               tensor->Size(), stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMemsetAsync failed with return value " << ret << ".";
      return ret;
    }
  }
  (void)SyncStream(stream_id);

  return 0;
}

TensorPtr AscendResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                            const std::vector<size_t> &before_padding_size,
                                                            const std::vector<size_t> &after_padding_size, size_t start,
                                                            size_t end) {
  if (start >= tensor_list.size() || end > tensor_list.size()) {
    MS_EXCEPTION(ValueError) << "start:" << start << ", end:" << end << ", but tensor_list size:" << tensor_list.size();
  }
  size_t size = std::accumulate(after_padding_size.begin() + start, after_padding_size.begin() + end - 1,
                                before_padding_size[end - 1]);
  ShapeVector shape = {int64_t(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = std::make_shared<tensor::Tensor>(tensor_list[start]->data_type(), shape);
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

TensorPtr AscendResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                         size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = std::make_shared<tensor::Tensor>(type_id, shape);
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

// ACL_EVENT_TIME_LINE: indicates that the number of created events is not limited, and the created events can be used
//  to compute the elapsed time between events, which may cause lost some performance.
// ACL_EVENT_SYNC: indicates that the number of created events is limited, and the created events can be used for
//  synchronization between multiple streams.
// ACL_EVENT_CAPTURE_STREAM_PROGRESS: indicates that the number of created events is not limited and high performance,
//  and the created events can not be used for timing and synchronization.
DeviceEventPtr AscendResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
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

DeviceEventPtr AscendResManager::CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) {
  auto flag = enable_timing ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
  auto event = std::make_shared<AscendEvent>(flag, use_extensional_api);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

void AscendResManager::MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor,
                              const std::string &to, bool blocking, bool *return_self) {
  device::MoveTo(src_tensor, dst_tensor, to, blocking, return_self);
}

bool AscendResManager::GetMemUceInfo(int32_t device_id) {
  aclrtMemUceInfo info[MAX_MEM_UCE_INFO_ARRAY_SIZE];
  size_t retSize = 0;
  auto ret = CALL_ASCEND_API(aclrtGetMemUceInfo, device_id, info, sizeof(info) / sizeof(aclrtMemUceInfo), &retSize);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Call aclrtGetMemUceInfo failed, ret code: " << ret;
    return false;
  }
  if (retSize == 0) {
    MS_LOG(WARNING) << "aclrtGetMemUceInfo get UCE size is 0.";
  }

  MS_LOG(INFO) << "aclrtGetMemUceInfo get UCE Error, retSize is " << retSize;

  MemUceInfo mem_uce_info;
  mem_uce_info.device_id = device_id;
  mem_uce_info.info.assign(info, info + retSize);
  mem_uce_info.retSize = retSize;

  std::lock_guard<std::mutex> lock(mem_uce_info_mutex_);
  mem_uce_info_ = mem_uce_info;

  return true;
}

std::vector<std::pair<device::DeviceMemPtr, size_t>> AscendResManager::GetMemUceAddr() {
  std::vector<std::pair<device::DeviceMemPtr, size_t>> mem_uce_addr;
  for (size_t i = 0; i < mem_uce_info_.info.size(); ++i) {
    std::pair<device::DeviceMemPtr, size_t> mem(mem_uce_info_.info[i].addr, mem_uce_info_.info[i].len);
    mem_uce_addr.emplace_back(mem);
  }
  MS_LOG(INFO) << "Get mem uce addr, size: " << mem_uce_addr.size();
  return mem_uce_addr;
}

void AscendResManager::UceMemRepair(int32_t device_id) {
  if (device_id != mem_uce_info_.device_id) {
    MS_LOG(EXCEPTION) << "Uce mem repair device id is not correct, device id is " << mem_uce_info_.device_id
                      << ", but got " << device_id << ".";
  }
  aclrtMemUceInfo *info = mem_uce_info_.info.data();
  auto ret = CALL_ASCEND_API(aclrtMemUceRepair, mem_uce_info_.device_id, info, mem_uce_info_.retSize);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtMemUceRepair failed, ret code: " << ret;
  }
  // Clear mem_uce_info.
  mem_uce_info_.device_id = 0;
  mem_uce_info_.info.clear();
  mem_uce_info_.retSize = 0;
}

void AscendResManager::StopDevice(int32_t device_id) {
  UCEException::GetInstance().set_force_stop_flag(true);
  // Wait 1 s to avoid stop device and suspension occur at the same time.
  const int64_t kTimeToWait = 1;
  std::this_thread::sleep_for(std::chrono::seconds(kTimeToWait));
  MS_LOG(INFO) << "Device id [" << device_id << "] stop device.";
  uint32_t timeout = 0;
  auto ret = CALL_ASCEND_API(aclrtDeviceTaskAbort, device_id, timeout);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtDeviceTaskAbort failed, ret code: " << ret;
  }
}

void *AscendResManager::GetCopyDataStream() const {
  auto copy_data_stream = AscendStreamMng::GetInstance().GetCopyStream();
  if (copy_data_stream == nullptr) {
    size_t copy_stream_id;
    AscendStreamMng::GetInstance().CreateStream(&copy_stream_id);
    MS_LOG(INFO) << "Create ascend copy data stream, stream id: " << copy_stream_id;
    copy_data_stream = AscendStreamMng::GetInstance().GetStream(copy_stream_id);
    AscendStreamMng::GetInstance().SetCopyStream(copy_data_stream);
  }
  return copy_data_stream;
}

bool AscendResManager::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                   const DeviceEventPtr &input_event) {
  return mem_manager_->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, input_event);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  return mem_manager_->WaitEvent(task_id_on_stream, user_stream_id, memory_stream_id);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) {
  return mem_manager_->WaitEvent(task_id_on_stream, user_stream_id);
}

bool AscendResManager::SyncAllEvents() { return mem_manager_->SyncAllEvents(); }

MS_REGISTER_HAL_RES_MANAGER(kAscendDevice, DeviceTargetType::kAscend, AscendResManager);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
