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
#include "plugin/cpu/res_manager/cpu_res_manager.h"
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "ir/tensor_new.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/memory_manager/memory_manager.h"

#include "mindspore/core/include/device_address/convert_tensor_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "plugin/cpu/res_manager/collective/ms_collective_comm_lib.h"
#endif

namespace mindspore {
namespace device {
namespace cpu {
void CPUResManager::Initialize() {
  mem_manager_ = std::make_shared<CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
}

void CPUResManager::Destroy() {
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

void *CPUResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void CPUResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void CPUResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                    const std::vector<size_t> &keep_addr_sizes) const {
  CPUMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

std::vector<void *> CPUResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                            uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocContinuousMemFromMemPool(size_list, stream_id);
}

std::pair<std::vector<size_t>, std::vector<size_t>> CPUResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  std::vector<size_t> after_padding_sizes = before_padding_sizes;
  auto stream_id = DefaultStream();
  auto device_ptr_list = AllocateContinuousMemory(before_padding_sizes, stream_id);
  for (size_t i = 0; i < after_padding_sizes.size(); ++i) {
    errno_t ret = memset_s(device_ptr_list[i], after_padding_sizes[i], 0, after_padding_sizes[i]);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Memset failed.";
    }
    MS_LOG(DEBUG) << "Clear ptr:" << device_ptr_list[i] << ", size:" << after_padding_sizes[i];
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // create device for all tensor in tensor list
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    const auto &ptr = device_ptr_list[i];
    auto device_address = CreateDeviceAddress(ptr, before_padding_sizes[i], tensor->shape(), Format::DEFAULT_FORMAT,
                                              tensor->data_type(), device_name, stream_id);
    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << ptr << ", size:" << before_padding_sizes[i]
                  << ", shape:" << tensor->shape() << ", data_type:" << TypeIdToString(tensor->data_type());
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    device::DeviceContextKey host_key = {device_address->GetDeviceType(), device_address->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->SyncAllStreams();
    DeviceAddressExtPtr src_ext = std::make_shared<DeviceAddressExt>(kernel::GetFormatFromStrToEnum(tensor->format()),
                                                                     tensor->data_type(), tensor->shape());
    DeviceAddressExtPtr dst_ext =
      std::make_shared<DeviceAddressExt>(Format::DEFAULT_FORMAT, tensor->data_type(), tensor->shape());
    SyncCopy(device_address, tensor->device_address(), device_address->stream_id(), src_ext, dst_ext);
    tensor->set_device_address(device_address);
  }
  return std::make_pair(before_padding_sizes, after_padding_sizes);
}

tensor::TensorPtr CPUResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                                 const std::vector<size_t> &before_padding_size,
                                                                 const std::vector<size_t> &after_padding_size,
                                                                 size_t start, size_t end) {
  if (start >= tensor_list.size() || end > tensor_list.size()) {
    MS_EXCEPTION(ValueError) << "start:" << start << ", end:" << end << ", but tensor_list size:" << tensor_list.size();
  }
  size_t size = std::accumulate(after_padding_size.begin() + start, after_padding_size.begin() + end - 1,
                                before_padding_size[end - 1]);
  ShapeVector shape = {int64_t(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = tensor::from_spec(tensor_list[start]->data_type(), shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(tensor_list[start]->device_address());
  auto ptr = tensor_list[start]->device_address()->GetMutablePtr();

  auto stream_id = DefaultStream();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address =
    CreateDeviceAddress(ptr, size, shape, Format::DEFAULT_FORMAT, tensor->data_type(), device_name, stream_id);
  tensor->set_device_address(device_address);
  return tensor;
}

tensor::TensorPtr CPUResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                              size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = tensor::from_spec(type_id, shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(first_tensor->device_address());
  auto ptr = first_tensor->device_address()->GetMutablePtr();
  auto offset_size = start * type_size;

  auto stream_id = DefaultStream();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address = CreateDeviceAddress(reinterpret_cast<uint8_t *>(ptr) + offset_size, tensor_size, shape,
                                            Format::DEFAULT_FORMAT, type_id, device_name, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
  tensor->set_device_address(device_address);
  return tensor;
}

DeviceAddressPtr CPUResManager::CreateDeviceAddress() const {
  auto device_address = std::make_shared<DeviceAddress>(nullptr, 0, kCPUDevice);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_name = device::DeviceType::kCPU;
  device_address->SetDeviceType(device_name);
  return device_address;
}

DeviceAddressPtr CPUResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                    const Format &format, TypeId type_id,
                                                    const std::string &device_name, uint32_t stream_id) const {
  auto device_address =
    std::make_shared<DeviceAddress>(ptr, size, shape_vector, format, type_id, kCPUDevice, stream_id);

  return device_address;
}

bool CPUResManager::SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                             size_t stream_id, const DeviceAddressExtPtr &src_ext,
                             const DeviceAddressExtPtr &dst_ext) const {
  return HostCopy(dst_device_sync, src_device_sync, src_ext, dst_ext);
}
bool CPUResManager::AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                              size_t stream_id, bool keep_src, const DeviceAddressExtPtr &src_ext,
                              const DeviceAddressExtPtr &dst_ext) const {
  return HostCopy(dst_device_sync, src_device_sync, src_ext, dst_ext);
}

bool CPUResManager::Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const {
  if (size == 0) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  auto ret_code = memcpy_s(dst, size, src, size);
  if (ret_code == ERANGE) {
    ConvertSameType(dst, src, size, kNumberTypeUInt8);
  } else if (ret_code != EOK) {
    MS_LOG(ERROR) << "Failed to copy tensor from ptr:" << src << " to :" << dst << " size:" << size;
    return false;
  }
  return true;
}

bool CPUResManager::CopyDirectly(void *dst, uint64_t size, const void *src, size_t stream_id, CopyType kind) const {
  return Copy(dst, src, size, kind, stream_id);
}

bool CPUResManager::LoadCollectiveCommLib() {
  bool using_mpi = common::UseMPI();
  if (using_mpi) {
    std::string mpi_comm_lib_name = "libmpi_collective.so";
    auto loader = std::make_shared<CollectiveCommLibLoader>(mpi_comm_lib_name);
    MS_EXCEPTION_IF_NULL(loader);
    if (!loader->Initialize()) {
      MS_LOG(EXCEPTION) << "Failed to load mpi collective library.";
    }

    void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

    auto instance_func = DlsymFuncObj(communication_lib_instance, collective_comm_lib_handle);
    collective_comm_lib_ = instance_func();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  } else {
#if defined(__linux__) && defined(WITH_BACKEND)
    collective_comm_lib_ = &MsCollectiveCommLib::GetInstance();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_);
#endif
  }
  return true;
}

CollectiveCommunicationLib *CPUResManager::collective_comm_lib() const { return collective_comm_lib_; }

MS_REGISTER_HAL_COPY_FUNC(
  DeviceType::kCPU,
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kCPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->SyncCopy(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }),
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id, bool keep_src,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kCPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->AsyncCopy(dst_device_sync, src_device_sync, stream_id, keep_src, src_ext,
                                                        dst_ext);
  }),
  ([](void *dst, const void *src, uint64_t size, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kCPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->Copy(dst, src, size, device::CopyType::kD2H, stream_id);
  }));

REGISTER_DEVICE_PTR_DELETER_MAKER(device::DeviceType::kCPU, ([](void *ptr, bool from_mem_pool) {
                                    if (ptr != nullptr && from_mem_pool) {
                                      CPUMemoryPool::GetInstance().FreeTensorMem(ptr);
                                    }
                                  }));
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
