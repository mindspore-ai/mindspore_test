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

#include "common/device_address.h"

namespace mindspore {
namespace device {
using ContinuousDeviceAddressesPtr = std::shared_ptr<std::vector<std::weak_ptr<DeviceAddress>>>;

DeviceAddress::DeviceAddress(const KernelTensorPtr &kernel_tensor)
    : kernel_tensor_(kernel_tensor), address_common_(kernel_tensor_->address_common()) {}

DeviceAddress::DeviceAddress(void *ptr, size_t size) {
  address_common_ = std::make_shared<AddressCommon>(ptr, size);
  kernel_tensor_ = std::make_shared<KernelTensor>();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id) {
  kernel_tensor_ = std::make_shared<KernelTensor>();
  address_common_ = kernel_tensor_->address_common();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->dtype_id_ = type_id;
  kernel_tensor_->SetStringFormat(format);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                             const KernelWithIndex &node_index)
    : node_index_(node_index) {
  kernel_tensor_ = std::make_shared<KernelTensor>();
  address_common_ = kernel_tensor_->address_common();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->dtype_id_ = type_id;
  kernel_tensor_->SetStringFormat(format);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id) {
  kernel_tensor_ = std::make_shared<KernelTensor>();
  address_common_ = kernel_tensor_->address_common();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  kernel_tensor_->set_device_id(device_id);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id,
                             const std::string &device_name, uint32_t device_id) {
  kernel_tensor_ = std::make_shared<KernelTensor>();
  address_common_ = kernel_tensor_->address_common();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  address_common_->dtype_id_ = type_id;
  kernel_tensor_->SetStringFormat(format);
  kernel_tensor_->set_device_id(device_id);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                             TypeId type_id, const std::string &device_name, uint32_t device_id, uint32_t stream_id) {
  address_common_ =
    std::make_shared<AddressCommon>(ptr, size, shape_vector, format, type_id, device_name, device_id, stream_id);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                             const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
    : node_index_(node_index) {
  kernel_tensor_ = std::make_shared<KernelTensor>();
  address_common_ = kernel_tensor_->address_common();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  address_common_->dtype_id_ = type_id;
  kernel_tensor_->SetStringFormat(format);
  kernel_tensor_->set_device_id(device_id);
}

DeviceAddress::~DeviceAddress() {
  if (IS_OUTPUT_ON(mindspore::kDebug) && address_common_ != nullptr && address_common_->pointer_ref_count_ != nullptr &&
      address_common_->pointer_ref_count_->new_ref_count() != SIZE_MAX && GetPtr() != nullptr) {
    MS_LOG(DEBUG) << "Maybe memory leak detect in device address:" << PrintInfo();
  }
  if (!from_mem_pool() && deleter_ && GetDevicePtr() != nullptr) {
    deleter_(static_cast<uint8_t *>(GetDevicePtr()));
    SetDevicePtr(nullptr);
  } else {
    address_common_->pointer_ref_count_ = nullptr;
  }
}

std::string DeviceAddress::PrintInfo() const {
  std::ostringstream ofs;
  ofs << this << " device type:" << GetDeviceType() << " kernel tensor:" << kernel_tensor_;
  if (kernel_tensor_ != nullptr) {
    ofs << " " << kernel_tensor_->PrintInfo();
  }
  ofs << " device address deleter:" << (deleter_ != nullptr) << " flag:" << flag_
      << " need sync user data:" << need_sync_user_data_ << " is view:" << is_view_;
  return ofs.str();
}

const KernelTensorPtr &DeviceAddress::kernel_tensor() const { return kernel_tensor_; }

void DeviceAddress::set_kernel_tensor(const KernelTensorPtr &kernel_tensor) {
  kernel_tensor_ = kernel_tensor;
  address_common_ = kernel_tensor_->address_common();
}

void DeviceAddress::set_device_synchronizer(const DeviceSynchronizerPtr &device_synchronizer) {
  MS_EXCEPTION_IF_NULL(kernel_tensor_);
  kernel_tensor_->set_device_synchronizer(device_synchronizer);
}

const void *DeviceAddress::GetPtr() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  return GetDevicePtr();
}

void DeviceAddress::set_ptr(void *ptr) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  address_common_->pointer_ref_count_->set_ptr(ptr);
  if (ptr != nullptr) {
    const auto &storage_info = GetStorageInfo();
    if (storage_info.host_ptr_ == nullptr && storage_info.file_name_.empty()) {
      status_ = DeviceAddressStatus::kInDevice;
    }
  }
}

size_t DeviceAddress::GetSize() const {
  auto kt = kernel_tensor();
  if (kt && kt->tensor_storage_info() && kt->tensor_storage_info()->is_contiguous &&
      (kt->tensor_storage_info()->ori_size != 0)) {
    return kt->tensor_storage_info()->ori_size;
  }
  return size();
}

void DeviceAddress::SetSize(size_t size) { address_common_->size_ = size; }

std::string DeviceAddress::format() const { return kernel::GetFormatFromEnumToStr(address_common_->format_); }

void DeviceAddress::set_format(const std::string &format) {
  address_common_->format_ = kernel::GetFormatFromStrToEnum(format);
}

const std::string &DeviceAddress::padding_type() const { return padding_type_; }

void DeviceAddress::set_padding_type(const std::string &padding_type) { padding_type_ = padding_type; }

TypeId DeviceAddress::type_id() const { return address_common_->dtype_id_; }

void DeviceAddress::set_type_id(TypeId type_id) { address_common_->dtype_id_ = type_id; }

bool DeviceAddress::from_mem_pool() const { return address_common_->pointer_ref_count_->from_mem_pool(); }

void DeviceAddress::set_from_mem_pool(bool from_mem_pool) const {
  address_common_->pointer_ref_count_->set_from_mem_pool(from_mem_pool);
}

void DeviceAddress::set_communication_ptr(uint8_t *communication_ptr) { MS_LOG(EXCEPTION) << "Not implemented error."; }

bool DeviceAddress::is_ptr_persisted() const { return address_common_->pointer_ref_count_->is_ptr_persisted(); }

void DeviceAddress::set_is_ptr_persisted(bool is_ptr_persisted) {
  address_common_->pointer_ref_count_->set_is_ptr_persisted(is_ptr_persisted);
}

void DeviceAddress::set_host_shape(const ShapeVector &shape) { kernel_tensor_->set_host_shape(shape); }

const ShapeVector &DeviceAddress::host_shape() const { return kernel_tensor_->host_shape(); }

void DeviceAddress::set_device_shape(const ShapeVector &shape) { device_shape_ = shape; }

const ShapeVector &DeviceAddress::device_shape() const { return device_shape_; }

bool DeviceAddress::from_persistent_mem() const { return from_persistent_mem_; }

void DeviceAddress::set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }

bool DeviceAddress::need_recycle() const { return need_recycle_; }

void DeviceAddress::set_need_recycle(bool need_recycle) { need_recycle_ = need_recycle; }

void DeviceAddress::set_status(DeviceAddressStatus status) { status_ = status; }

DeviceAddressStatus DeviceAddress::status() const { return status_; }

DeviceType DeviceAddress::GetDeviceType() const { return DeviceType::kUnknown; }

void *DeviceAddress::GetMutablePtr() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  return GetDevicePtr();
}

const ShapeVector &DeviceAddress::GetShapeVector() const { return address_common_->shape_vector_; }

const TensorStorageInfoPtr DeviceAddress::GetTensorStorageInfo() const {
  if (address_common_ == nullptr) {
    return nullptr;
  }

  return address_common_->tensor_storage_info_;
}

void DeviceAddress::set_tensor_storage_info(const TensorStorageInfoPtr &tensor_storage_info) {
  address_common_->tensor_storage_info_ = tensor_storage_info;
}

const std::string &DeviceAddress::device_name() const { return address_common_->device_name_; }

uint32_t DeviceAddress::device_id() const { return address_common_->device_id_; }

void DeviceAddress::set_stream_id(uint32_t stream_id) { address_common_->stream_id_ = stream_id; }

const uint32_t DeviceAddress::stream_id() const { return address_common_->stream_id_; }

void DeviceAddress::AddHeldByNode(const std::weak_ptr<ValueNode> &value_node) {
  (void)held_by_nodes_.emplace_back(value_node);
}

std::vector<std::weak_ptr<ValueNode>> DeviceAddress::held_by_nodes() const { return held_by_nodes_; }

void DeviceAddress::ClearHeldByNodes() { held_by_nodes_.clear(); }

void DeviceAddress::SetNodeIndex(const AnfNodePtr &node, size_t out_index) { node_index_ = {node, out_index}; }

KernelWithIndex DeviceAddress::GetNodeIndex() const {
  return node_index_.first.expired() ? KernelWithIndex{nullptr, node_index_.second}
                                     : KernelWithIndex{node_index_.first.lock(), node_index_.second};
}

size_t DeviceAddress::IncreaseCounter() { return address_common_->pointer_ref_count_->IncreaseCounter(); }

size_t DeviceAddress::DecreaseCounter() { return address_common_->pointer_ref_count_->DecreaseCounter(); }

void DeviceAddress::IncreaseNewRefCount(size_t i) {
  address_common_->pointer_ref_count_->IncreaseNewRefCount(i);
  MS_LOG(DEBUG) << "Increase new ref count for device address:" << PrintInfo();
}

size_t DeviceAddress::DecreaseNewRefCount() {
  size_t ref_count = address_common_->pointer_ref_count_->DecreaseNewRefCount();
  MS_LOG(DEBUG) << "Decrease new ref count for device address:" << PrintInfo();
  return ref_count;
}

void DeviceAddress::set_new_ref_count(size_t new_ref_count) const {
  address_common_->pointer_ref_count_->set_new_ref_count(new_ref_count);
}

size_t DeviceAddress::new_ref_count() const { return address_common_->pointer_ref_count_->new_ref_count(); }

void DeviceAddress::set_original_ref_count(size_t original_ref_count) const {
  address_common_->pointer_ref_count_->set_original_ref_count(original_ref_count);
}

size_t DeviceAddress::original_ref_count() const { return address_common_->pointer_ref_count_->original_ref_count(); }

void DeviceAddress::set_ref_count(size_t ref_count) const {
  address_common_->pointer_ref_count_->set_ref_count(ref_count);
}

size_t DeviceAddress::ref_count() const { return address_common_->pointer_ref_count_->ref_count(); }

void DeviceAddress::ResetRefCount() { address_common_->pointer_ref_count_->ResetRefCount(); }

void DeviceAddress::IncreaseOriginalRefCount() {
  if (original_ref_count() < SIZE_MAX) {
    address_common_->pointer_ref_count_->IncreaseOriginalRefCount();
  }
}

void DeviceAddress::DecreaseOriginalRefCount() {
  if ((original_ref_count() < SIZE_MAX) && (original_ref_count() > 0)) {
    address_common_->pointer_ref_count_->DecreaseOriginalRefCount();
  }
}

void DeviceAddress::IncreaseRefCount(size_t increase_cnt) {
  address_common_->pointer_ref_count_->IncreaseRefCount(increase_cnt);
}

size_t DeviceAddress::DecreaseRefCount() { return address_common_->pointer_ref_count_->DecreaseRefCount(); }

void DeviceAddress::set_dynamic_ref_count(int32_t dynamic_ref_count) {
  address_common_->pointer_ref_count_->set_dynamic_ref_count(dynamic_ref_count);
}

int32_t DeviceAddress::dynamic_ref_count() const { return address_common_->pointer_ref_count_->dynamic_ref_count(); }

void DeviceAddress::IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt) {
  address_common_->pointer_ref_count_->IncreaseDynamicRefCount(op_object, increase_cnt);
}

void DeviceAddress::IncreaseDynamicRefCount(const std::string &op_object) {
  address_common_->pointer_ref_count_->IncreaseDynamicRefCount(op_object);
}

int32_t DeviceAddress::DecreaseDynamicRefCount(const std::string &op_object) {
  return address_common_->pointer_ref_count_->DecreaseDynamicRefCount(op_object);
}

bool DeviceAddress::IsPtrValid() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  return GetDevicePtr() != nullptr;
}

bool DeviceAddress::IsNotNeedAlloc() const {
  return IsPtrValid() || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed);
}

// Return the valid device ptr.
void *DeviceAddress::GetValidPtr(size_t) {
  if (user_data() == nullptr || (!need_sync_user_data_)) {
    return GetDevicePtr();
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (!need_sync_user_data_) {
    return GetDevicePtr();
  }
  auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
  if (sync_handler == nullptr) {
    MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
    return GetDevicePtr();
  }
  (*sync_handler)(this);
  need_sync_user_data_ = false;
  return GetDevicePtr();
}

void DeviceAddress::Swap(DeviceAddress *other) {
  MS_EXCEPTION_IF_NULL(other);
  if (other == this) {
    return;
  }
  other->SetDevicePtr(GetDevicePtr());

  other->set_from_mem_pool(this->from_mem_pool());
  other->set_deleter(deleter());
  other->set_need_sync_user_data(need_sync_user_data_);
  SetDevicePtr(nullptr);
  this->set_from_mem_pool(false);
  deleter_ = nullptr;
  kernel_tensor()->set_task_id_on_stream(other->kernel_tensor()->task_id_on_stream());
  kernel_tensor()->set_managed_by_somas(other->kernel_tensor()->managed_by_somas());
}

const UserDataPtr &DeviceAddress::user_data() const { return kernel_tensor_->user_data(); }

void DeviceAddress::set_user_data(const UserDataPtr &user_data) { kernel_tensor_->set_user_data(user_data); }

size_t DeviceAddress::flag() const { return flag_; }

void DeviceAddress::set_flag(size_t flag) { flag_ = flag; }

void DeviceAddress::UpdateFlag(size_t flag) { SET_FLAG(flag_, flag); }

void DeviceAddress::ClearFlag(size_t flag) { CLEAR_FLAG(flag_, flag); }

std::pair<AnfNodeWeakPtr, size_t> DeviceAddress::node_index() const { return node_index_; }

void DeviceAddress::set_deleter(const std::function<void(uint8_t *)> &deleter) { deleter_ = deleter; }

std::function<void(uint8_t *)> DeviceAddress::deleter() const { return deleter_; }

bool DeviceAddress::need_sync_user_data() { return need_sync_user_data_; }

void DeviceAddress::set_need_sync_user_data(bool need_sync_user_data) { need_sync_user_data_ = need_sync_user_data; }

const PointerRefCountPtr &DeviceAddress::pointer_ref_count() const { return address_common_->pointer_ref_count_; }

void DeviceAddress::set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt) {
  MS_EXCEPTION_IF_NULL(ptr_ref_cnt);
  address_common_->pointer_ref_count_ = ptr_ref_cnt;
}

void DeviceAddress::set_is_view(bool is_view) { is_view_ = is_view; }

bool DeviceAddress::is_view() const { return is_view_; }

AddressCommonPtr DeviceAddress::address_common() const { return address_common_; }

ContinuousDeviceAddressesPtr DeviceAddress::continuous_device_addresses() const { return continuous_device_addresses_; }

void DeviceAddress::set_continuous_device_addresses(const ContinuousDeviceAddressesPtr &continuous_device_addresses) {
  continuous_device_addresses_ = continuous_device_addresses;
}
}  // namespace device
}  // namespace mindspore
