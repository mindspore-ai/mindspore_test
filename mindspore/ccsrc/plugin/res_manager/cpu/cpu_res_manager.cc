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
#include "plugin/res_manager/cpu/cpu_res_manager.h"
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "utils/ms_context.h"
#include "runtime/device/res_manager/memory_manager.h"
#include "plugin/res_manager/cpu/cpu_mem_manager/cpu_hash_table_util.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "runtime/device/res_manager/tensor_array.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"

#include "ir/tensor_api.h"
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
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    device::ResKey res_key{device_address->GetDeviceType(), device_address->device_id()};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    res_manager->SyncAllStreams();
    SyncCopy(device_address, tensor->device_address(), device_address->stream_id());
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
  auto tensor = tensor::empty(tensor_list[start]->data_type(), shape, device::DeviceType::kNone);
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

tensor::TensorPtr CPUResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                              size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = tensor::empty(type_id, shape, device::DeviceType::kNone);
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

namespace {
// Create user data content(such as CPU hash table) and set user data reference into device_address.
void FillUserData(const UserDataPtr &user_data, DeviceAddress *device_address) {
  MS_EXCEPTION_IF_NULL(user_data);
  MS_EXCEPTION_IF_NULL(device_address);

  // Save reference of user data in device address.
  device_address->set_user_data(user_data);

  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  if (user_data_type == nullptr) {
    return;
  }
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = cpu_hash_table_funcs.find({*key_type, *value_type});
    if (iter != cpu_hash_table_funcs.end()) {
      // Create CPU hash table and set into `user_data`.
      return std::get<kCreateFuncIndex>(iter->second)(user_data);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type, key type:" << TypeIdLabel(*key_type)
                        << ", value type:" << TypeIdLabel(*value_type);
    }
  } else {
    MS_LOG(EXCEPTION) << "Invalid user data type:" << *user_data_type;
  }
}
}  // namespace

DeviceAddressPtr CPUResManager::CreateDeviceAddress() const {
  auto device_address = std::make_shared<CPUDeviceAddress>();
  device_address->set_device_name(GetDeviceNameByType(res_key_.device_name_));
  device_address->set_device_id(res_key_.device_id_);
  return device_address;
}

DeviceAddressPtr CPUResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                    const Format &format, TypeId type_id,
                                                    const std::string &device_name, uint32_t device_id,
                                                    uint32_t stream_id, const UserDataPtr &user_data) const {
  auto real_device_name = device_name;
  auto real_device_id = device_id;
  if (device_name.empty()) {
    real_device_name = GetDeviceNameByType(res_key_.device_name_);
    real_device_id = res_key_.device_id_;
    MS_LOG(DEBUG) << "Create device address with real device name: " << real_device_name
                  << ", real device id: " << real_device_id;
  }
  auto device_address = std::make_shared<CPUDeviceAddress>(ptr, size, shape_vector, format, type_id, real_device_name,
                                                           real_device_id, stream_id);

  if (user_data != nullptr) {
    FillUserData(user_data, device_address.get());
  }

  return device_address;
}

bool CPUResManager::SyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                             size_t stream_id) const {
  return AsyncCopy(dst_device_sync, src_device_sync, stream_id);
}
bool CPUResManager::AsyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                              size_t stream_id) const {
  const auto &dst_device_address = dynamic_cast<const CPUDeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const CPUDeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(INFO) << "No need sync for dst device address: " << dst_device_address
                 << " and src device address: " << src_device_address;
    return true;
  }

  if (dst_device_address->format() != src_device_address->format()) {
    MS_LOG(ERROR) << "Format is different, src(format:" << src_device_address->format()
                  << "), dst(format:" << dst_device_address->format() << ") for device address:" << dst_device_address;
    return false;
  }
  auto dst_ptr = dst_device_address->GetMutablePtr();
  auto src_ptr = src_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_device_address->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetMutablePtr());
  if (dst_ptr == src_ptr) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }
  auto dst_type_id = dst_device_address->type_id();
  auto src_type_id = src_device_address->type_id();

  if (src_type_id == dst_type_id) {
    if (src_device_address->GetSize() < dst_device_address->GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, src size: " << src_device_address->GetSize()
                      << ", dst size: " << dst_device_address->GetSize();
      return true;
    }
    auto ret_code = memcpy_s(dst_ptr, src_device_address->GetSize(), src_ptr, dst_device_address->GetSize());
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      ConvertSameType(dst_device_address->GetMutablePtr(), src_device_address->GetMutablePtr(),
                      dst_device_address->GetSize(), src_type_id);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor from device address:" << src_device_address
                    << " to :" << dst_device_address;
      return false;
    } else {
      return true;
    }
  }

  if (dst_type_id == kNumberTypeFloat16 && src_type_id == kNumberTypeFloat32) {
    FloatToHalf(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeFloat64 && src_type_id == kNumberTypeFloat32) {
    FloatToDouble(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(double));
  } else if (dst_type_id == kNumberTypeFloat32 && src_type_id == kNumberTypeFloat64) {
    DoubleToFloat(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeInt16 && src_type_id == kNumberTypeInt32) {
    IntToShort(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeInt64 && src_type_id == kNumberTypeInt32) {
    IntToLong(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. src type: " << TypeIdLabel(src_type_id)
                  << ", dst type: " << TypeIdLabel(dst_type_id) << " device_address:" << dst_device_address << " !";
    return false;
  }
  return true;
}

bool CPUResManager::Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const {
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

MS_REGISTER_HAL_COPY_FUNC(
  DeviceType::kCPU, ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{DeviceType::kCPU, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    return res_manager->SyncCopy(dst_device_sync, src_device_sync, stream_id);
  }),
  ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{DeviceType::kCPU, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    return res_manager->SyncCopy(dst_device_sync, src_device_sync, stream_id);
  }));

MS_REGISTER_HAL_RES_MANAGER(kCPUDevice, DeviceType::kCPU, CPUResManager);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
