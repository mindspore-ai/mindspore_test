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
#include "plugin/res_manager/gpu/gpu_res_manager.h"
#include <libgen.h>
#include <cuda.h>
#include <utility>
#include <vector>
#include <string>
#include "plugin/res_manager/gpu/device/gpu_memory_manager.h"
#include "plugin/res_manager/gpu/device_context_conf/op_precision_conf.h"
#include "plugin/res_manager/gpu/device_context_conf/op_tuning_conf.h"
#include "plugin/res_manager/gpu/device/gpu_device_manager.h"
#include "plugin/res_manager/gpu/device/gpu_pin_mem_pool.h"
#include "plugin/res_manager/gpu/device/gpu_device_address.h"
#include "plugin/res_manager/gpu/device/gpu_device_synchronizer.h"
#include "plugin/res_manager/gpu/device/gpu_event.h"
#include "plugin/res_manager/gpu/device/gpu_hash_table_util.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/device/tensor_array.h"

namespace mindspore {
namespace device {
namespace gpu {
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

void GPUResManager::Initialize() {
  // Set device id
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    res_key_.device_id_ = distributed::collective::CollectiveManager::instance()->local_rank_id();

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, res_key_.device_id_);
  }

  MS_LOG(INFO) << "Set GPU device id index " << res_key_.device_id_;
  // Set device id and initialize device resource.
  if (!InitDevice()) {
    MS_LOG(EXCEPTION) << "GPU InitDevice failed.";
  }

  // Initialize memory pool.
  mem_manager_ = std::make_shared<GPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    swap_manager_ = std::make_shared<SwapManager>(GPUDeviceManager::GetInstance().default_stream_id(),
                                                  &GPUMemoryAllocator::GetInstance(), &GPUPinMemPool::GetInstance());
  }

  // Initialize NCCL.
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
#if defined(_WIN32)
    MS_LOG(EXCEPTION) << "windows not support nccl.";
#endif
  }
}

namespace {
float GetCudaCap(const uint32_t device_id) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  auto major = prop.major;
  auto minor = prop.minor;
  return static_cast<float>(major * 10 + minor) / 10.0;
}
}  // namespace

bool GPUResManager::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }

  if (!GPUDeviceManager::GetInstance().is_device_id_init()) {
    if (!GPUDeviceManager::GetInstance().set_cur_device_id(res_key_.device_id_)) {
      MS_LOG(ERROR) << "Failed to set current device id: " << SizeToInt(res_key_.device_id_);
      return false;
    }
  }
  // Check the Cuda capability
  const float cuda_cap = GetCudaCap(res_key_.device_id_);
  if (cuda_cap < 5.3) {
    MS_LOG(WARNING) << "The device with Cuda compute capability " << cuda_cap
                    << " is lower than the minimum required capability " << 5.3
                    << ", this may cause some unexpected problems and severely affect the results. "
                    << "Eg: the outputs are all zeros.\n"
                    << "Device with a compute capability > " << 5.3 << " is required, "
                    << "and it is recommended to use devices with a compute capability >= " << 7;
  }

  // Initialize device resource, such as stream, cudnn and cublas handle.
  GPUDeviceManager::GetInstance().InitDevice();
  return true;
}

void GPUResManager::Destroy() {
  (void)DestroyAllEvents();
  if (DataQueueMgr::GetInstance().IsInit()) {
    if (!DataQueueMgr::GetInstance().IsClosed() && !DataQueueMgr::GetInstance().CloseNotify()) {
      MS_LOG(ERROR) << "Could not close gpu data queue.";
    }
    DataQueueMgr::GetInstance().Release();
  }

  // Release stream, cudnn and cublas handle, etc.
  GPUDeviceManager::GetInstance().ReleaseDevice();

  // Release device memory
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}
void *GPUResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (!BindDeviceToCurrentThread(false)) {
    return nullptr;
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

size_t GPUResManager::GetAvailableMemSize() const { return mem_manager_->GetAvailableMemSize(); }

void GPUResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_EXCEPTION_IF_NULL(ptr);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void GPUResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                    const std::vector<size_t> &keep_addr_sizes) const {
  GPUMemoryAllocator::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

bool GPUResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (device_name_in_address != GetDeviceNameByType(res_key_.device_name_)) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << GetDeviceNameByType(res_key_.device_name_);
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (!BindDeviceToCurrentThread(false)) {
    return false;
  }

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  void *device_ptr;
  if (swap_manager_ != nullptr) {
    device_ptr = swap_manager_->AllocDeviceMemory(address->GetSize(), stream_id);
  } else {
    device_ptr =
      mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(), false, stream_id);
  }
  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  return true;
}

std::vector<void *> GPUResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                            uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    std::vector<void *> ptr_list;
    return ptr_list;
  }

  // Memory allocation ensures memory alignment.
  std::vector<size_t> align_size_list;
  for (size_t size : size_list) {
    auto align_size = GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    (void)align_size_list.emplace_back(align_size);
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceContinuousMem(align_size_list, stream_id);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(align_size_list, stream_id);
}

std::pair<std::vector<size_t>, std::vector<size_t>> GPUResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  std::vector<size_t> after_padding_sizes;
  for (auto &size : before_padding_sizes) {
    auto align_size = GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    after_padding_sizes.emplace_back(align_size);
  }
  auto stream_id = DefaultStream();
  auto device_ptr_list = AllocateContinuousMemory(before_padding_sizes, stream_id);
  for (size_t i = 0; i < after_padding_sizes.size(); ++i) {
    auto ret = cudaMemset(device_ptr_list[i], 0, after_padding_sizes[i]);
    if (ret != cudaSuccess) {
      MS_LOG(EXCEPTION) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
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
    auto device_address = CreateDeviceAddress(ptr, before_padding_sizes[i], tensor->shape(), Format::DEFAULT_FORMAT,
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

tensor::TensorPtr GPUResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                                 const std::vector<size_t> &before_padding_size,
                                                                 const std::vector<size_t> &after_padding_size,
                                                                 size_t start, size_t end) {
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

  auto device_address = CreateDeviceAddress(ptr, size, shape, Format::DEFAULT_FORMAT, tensor->data_type(), device_name,
                                            device_id, stream_id);
  tensor->set_device_address(device_address);
  return tensor;
}

tensor::TensorPtr GPUResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
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
                                            Format::DEFAULT_FORMAT, type_id, device_name, device_id, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
  tensor->set_device_address(device_address);
  return tensor;
}

// Relevant function to manage memory statistics
size_t GPUResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}
size_t GPUResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}
size_t GPUResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}
size_t GPUResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}
size_t GPUResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}
size_t GPUResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}
std::unordered_map<std::string, std::size_t> GPUResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}
std::unordered_map<std::string, std::size_t> GPUResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GPUResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GPUResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}
void GPUResManager::ResetMaxMemoryReserved() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryReserved();
}
void GPUResManager::ResetMaxMemoryAllocated() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryAllocated();
}

namespace {
// Create data in user data for device address.
void SetUserData(DeviceAddress *device_address, const UserDataPtr &user_data) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(user_data);

  device_address->set_user_data(user_data);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  if (user_data_type == nullptr) {
    return;
  }
  MS_LOG(EXCEPTION) << "Invalid user data type:" << *user_data_type;
}
}  // namespace

DeviceAddressPtr GPUResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->device_name().empty()) {
    kernel_tensor->set_device_name(GetDeviceNameByType(res_key_.device_name_));
    kernel_tensor->set_device_id(res_key_.device_id_);
  }
  auto device_address = std::make_shared<GPUDeviceAddress>(kernel_tensor);

  const auto &user_data = kernel_tensor->user_data();
  if (user_data != nullptr) {
    SetUserData(device_address.get(), user_data);
  }

  device_address->set_device_synchronizer(std::make_shared<GPUDeviceSynchronizer>());
  return device_address;
}

DeviceAddressPtr GPUResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                    const Format &format, TypeId type_id,
                                                    const std::string &device_name, uint32_t device_id,
                                                    uint32_t stream_id) const {
  return std::make_shared<GPUDeviceAddress>(ptr, size, shape_vector, format, type_id, device_name, device_id,
                                            stream_id);
}

bool GPUResManager::CreateStream(size_t *stream_id) const {
  return GPUDeviceManager::GetInstance().CreateStream(stream_id);
}

bool GPUResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  return GPUDeviceManager::GetInstance().CreateStreamWithPriority(stream_id, priority);
}

size_t GPUResManager::QueryStreamSize() const { return GPUDeviceManager::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> GPUResManager::GetStreamIds() const { return GPUDeviceManager::GetInstance().GetStreamIds(); }

bool GPUResManager::single_op_multi_stream_enable() const {
  return GPUDeviceManager::GetInstance().single_op_multi_stream_enable();
}

void GPUResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return GPUDeviceManager::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *GPUResManager::GetStream(size_t stream_id) const { return GPUDeviceManager::GetInstance().GetStream(stream_id); }

size_t GPUResManager::GetCommunicationStreamID() const {
  MS_LOG(WARNING) << "CommunicationStreamID is no create yet, return default stream.";
  return GPUDeviceManager::GetInstance().default_stream_id();
}

bool GPUResManager::DestroyStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().DestroyStream(stream_id);
}

void GPUResManager::SetCurrentStreamId(size_t stream_id) {
  GPUDeviceManager::GetInstance().set_current_stream(stream_id);
}

size_t GPUResManager::GetCurrentStreamId() const { return GPUDeviceManager::GetInstance().current_stream(); }

bool GPUResManager::QueryStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().QueryStream(stream_id);
}

bool GPUResManager::SyncStream(size_t stream_id) const { return GPUDeviceManager::GetInstance().SyncStream(stream_id); }

bool GPUResManager::SyncAllStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Fail to bind device to current thread";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncAllStreams();
}
bool GPUResManager::SyncNotDefaultStreams() const { return GPUDeviceManager::GetInstance().SyncNotDefaultStreams(); }

size_t GPUResManager::DefaultStream() const { return GPUDeviceManager::GetInstance().default_stream_id(); }

// cudaEventRecordDefault 0x0 | cudaEventRecordExternal 0x1 | cudaEventWaitExternal 0x1, no need to set again.
DeviceEventPtr GPUResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }
  uint32_t flag = cudaEventDefault;
  flag |= cudaEventDisableTiming;
  flag |= cudaEventBlockingSync;
  return std::make_shared<GpuEvent>(flag);
}

DeviceEventPtr GPUResManager::CreateEventWithFlag(bool enable_timing, bool blocking, bool) {
  uint32_t flag =
    (blocking ? cudaEventBlockingSync : cudaEventDefault) | (enable_timing ? cudaEventDefault : cudaEventDisableTiming);
  auto event = std::make_shared<GpuEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

bool GPUResManager::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (!event->DestroyEvent()) {
    MS_LOG(ERROR) << "DestroyEvent failed.";
    return false;
  }

  std::lock_guard<std::mutex> lock(device_events_mutex_);
  const auto &iter = std::find(device_events_.begin(), device_events_.end(), event);
  if (iter == device_events_.end()) {
    MS_LOG(ERROR) << "Can't find specified device event.";
    return false;
  }
  (void)device_events_.erase(iter);
  return true;
}

bool GPUResManager::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    // Reduce the scopt to prevent deadlock.
    std::lock_guard<std::mutex> lock(device_events_mutex_);
    device_events_inner = device_events_;
    device_events_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    MS_EXCEPTION_IF_NULL(event);
    if (!event->DestroyEvent()) {
      MS_LOG(ERROR) << "DestroyEvent failed.";
    }
  });
  device_events_.clear();
  return true;
}

bool GPUResManager::LoadCollectiveCommLib() {
#ifdef ENABLE_MPI
  std::string nvidia_comm_lib_name = GetCurrentDir() + "/gpu" + std::to_string(CUDA_VERSION / 1000) + "." +
                                     std::to_string(CUDA_VERSION / 10 % 10) + "/libnvidia_collective.so";
  auto loader = std::make_shared<CollectiveCommLibLoader>(nvidia_comm_lib_name);
  MS_EXCEPTION_IF_NULL(loader);
  if (!loader->Initialize()) {
    MS_LOG(EXCEPTION) << "Loading NCCL collective library failed.";
    return false;
  }
  void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

  auto instance_func = DlsymFuncObj(communication_lib_instance, collective_comm_lib_handle);
  collective_comm_lib_ = instance_func();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  return true;
#else
  return false;
#endif
}

static thread_local bool cur_thread_device_inited{false};

bool GPUResManager::BindDeviceToCurrentThread(bool force_bind) const {
  if (cur_thread_device_inited && !force_bind) {
    return true;
  }

  if (!CudaDriver::SetDevice(UintToInt(res_key_.device_id_))) {
    MS_LOG(ERROR) << "Failed to set device id: " << res_key_.device_id_;
    return false;
  }

  cur_thread_device_inited = true;
  return true;
}

std::shared_ptr<void> GPUResManager::AllocateHostMemory(size_t size) const {
  void *ptr;
  if (CudaDriver::AllocHostPinnedMem(size, &ptr) != size) {
    MS_LOG(ERROR) << "Failed to allow host pinned memory.";
    return nullptr;
  }

  return std::shared_ptr<void>(ptr, [](void *ptr) -> void {
    if (ptr != nullptr) {
      CudaDriver::FreeHostPinnedMem(ptr);
    }
  });
}

MS_REGISTER_HAL_RES_MANAGER(kGPUDevice, DeviceType::kGPU, GPUResManager);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
