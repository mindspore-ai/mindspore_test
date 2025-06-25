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
#include "common/format_utils.h"

namespace mindspore {
namespace device {
using ContinuousDeviceAddressesPtr = std::shared_ptr<std::vector<std::weak_ptr<DeviceAddress>>>;

DeviceAddress::DeviceAddress() { address_common_ = std::make_shared<AddressCommon>(); }
DeviceAddress::DeviceAddress(const AddressCommonPtr &address_common) : address_common_(address_common) {}

DeviceAddress::DeviceAddress(void *ptr, size_t size) { address_common_ = std::make_shared<AddressCommon>(ptr, size); }

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id) {
  address_common_ = std::make_shared<AddressCommon>();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->dtype_id_ = type_id;
  address_common_->format_ = kernel::GetFormatFromStrToEnum(format);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                             const KernelWithIndex &node_index)
    : node_index_(node_index) {
  address_common_ = std::make_shared<AddressCommon>();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->dtype_id_ = type_id;
  address_common_->format_ = kernel::GetFormatFromStrToEnum(format);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id) {
  address_common_ = std::make_shared<AddressCommon>();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  address_common_->device_id_ = device_id;
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id,
                             const std::string &device_name, uint32_t device_id) {
  address_common_ = std::make_shared<AddressCommon>();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  address_common_->dtype_id_ = type_id;
  address_common_->format_ = kernel::GetFormatFromStrToEnum(format);
  address_common_->device_id_ = device_id;
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                             TypeId type_id, const std::string &device_name, uint32_t device_id, uint32_t stream_id) {
  address_common_ =
    std::make_shared<AddressCommon>(ptr, size, shape_vector, format, type_id, device_name, device_id, stream_id);
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                             const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
    : node_index_(node_index) {
  address_common_ = std::make_shared<AddressCommon>();
  address_common_->pointer_ref_count_->set_ptr(ptr);
  address_common_->size_ = size;
  address_common_->device_name_ = device_name;
  address_common_->dtype_id_ = type_id;
  address_common_->format_ = kernel::GetFormatFromStrToEnum(format);
  address_common_->device_id_ = device_id;
}

DeviceAddress::~DeviceAddress() {
  if (IS_OUTPUT_ON(mindspore::kDebug) && address_common_ != nullptr && address_common_->pointer_ref_count_ != nullptr &&
      address_common_->pointer_ref_count_->new_ref_count() != SIZE_MAX && GetPtr() != nullptr) {
    MS_LOG(DEBUG) << "Maybe memory leak detect in device address:" << ToString();
  }
  if (!from_mem_pool() && deleter_ && GetDevicePtr() != nullptr) {
    deleter_(static_cast<uint8_t *>(GetDevicePtr()));
    SetDevicePtr(nullptr);
  } else {
    address_common_->pointer_ref_count_ = nullptr;
  }
}

std::string DeviceAddress::ToString() const {
  std::ostringstream ofs;
  ofs << this << " device type:" << GetDeviceType() << " address common:" << address_common_;
  if (address_common_ != nullptr) {
    ofs << " " << address_common_->ToString();
  }
  ofs << " device address deleter:" << (deleter_ != nullptr) << " flag:" << flag_
      << " need sync user data:" << need_sync_user_data_ << " user data:" << user_data_ << " is view:" << is_view_;
  return ofs.str();
}

void DeviceAddress::CloneDeviceAddress(const DeviceAddressPtr &device_address) {
  device_address->set_address_common(std::make_shared<AddressCommon>(*address_common_));
  device_address->set_device_shape(device_shape_);
  device_address->set_from_persistent_mem(from_persistent_mem_);
  device_address->set_need_recycle(need_recycle_);
  device_address->set_padding_type(padding_type_);
  device_address->set_flag(flag_);
  device_address->set_is_view(is_view_);
  device_address->set_status(status_);
  device_address->set_deleter(deleter_);
  device_address->set_continuous_device_addresses(continuous_device_addresses_);
  device_address->set_user_data(user_data_);
  device_address->set_need_sync_user_data(need_sync_user_data_);
  device_address->set_host_shape(host_shape_);
  device_address->set_heterogeneous_info(hete_info_);
  auto node_with_index = GetNodeIndex();
  device_address->SetNodeIndex(node_with_index.first, node_with_index.second);
  for (const auto &held_by_node : held_by_nodes_) {
    device_address->AddHeldByNode(held_by_node);
  }
}

const void *DeviceAddress::GetPtr() const { return GetDevicePtr(); }

void DeviceAddress::set_ptr(void *ptr) {
  address_common_->pointer_ref_count_->set_ptr(ptr);
  if (ptr != nullptr) {
    status_ = DeviceAddressStatus::kInDevice;
  }
}

size_t DeviceAddress::GetSize() const {
  if (address_common_ && address_common_->tensor_storage_info_ &&
      (address_common_->tensor_storage_info_->ori_size != 0)) {
    return address_common_->tensor_storage_info_->ori_size;
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

void DeviceAddress::set_device_shape(const ShapeVector &shape) { device_shape_ = shape; }

const ShapeVector &DeviceAddress::device_shape() const { return device_shape_; }

bool DeviceAddress::from_persistent_mem() const { return from_persistent_mem_; }

void DeviceAddress::set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }

bool DeviceAddress::need_recycle() const { return need_recycle_; }

void DeviceAddress::set_need_recycle(bool need_recycle) { need_recycle_ = need_recycle; }

void DeviceAddress::set_status(DeviceAddressStatus status) { status_ = status; }

DeviceAddressStatus DeviceAddress::status() const { return status_; }

DeviceType DeviceAddress::GetDeviceType() const { return DeviceType::kUnknown; }

void *DeviceAddress::GetMutablePtr() const { return GetDevicePtr(); }

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
void DeviceAddress::set_device_name(const std::string &device_name) { address_common_->device_name_ = device_name; }

uint32_t DeviceAddress::device_id() const { return address_common_->device_id_; }
void DeviceAddress::set_device_id(uint32_t device_id) { address_common_->device_id_ = device_id; }

void DeviceAddress::set_stream_id(uint32_t stream_id) { address_common_->stream_id_ = stream_id; }

const uint32_t DeviceAddress::stream_id() const { return address_common_->stream_id_; }

bool DeviceAddress::managed_by_somas() const { return address_common_->managed_by_somas_; }

void DeviceAddress::set_managed_by_somas(bool managed_by_somas) {
  address_common_->managed_by_somas_ = managed_by_somas;
}

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

void DeviceAddress::IncreaseNewRefCount(const std::string &op_name, size_t i) {
  address_common_->pointer_ref_count_->IncreaseNewRefCount(i);
  MS_LOG(DEBUG) << "Op:" << op_name << " increase new ref count for device address:" << ToString();
}

void DeviceAddress::IncreaseNewRefCount(size_t i) { address_common_->pointer_ref_count_->IncreaseNewRefCount(i); }

size_t DeviceAddress::DecreaseNewRefCount(const std::string &op_name) {
  size_t ref_count = address_common_->pointer_ref_count_->DecreaseNewRefCount();
  MS_LOG(DEBUG) << "Op:" << op_name << " decrease new ref count for device address:" << ToString();
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

void DeviceAddress::set_ref_count_without_hold(const PointerRefCountPtr &ptr_ref_cnt) {
  if (ptr_ref_cnt == nullptr || address_common_ == nullptr || address_common_->pointer_ref_count_ == nullptr) {
    return;
  }
  address_common_->pointer_ref_count_->set_ptr(ptr_ref_cnt->ptr());
  address_common_->pointer_ref_count_->set_from_mem_pool(ptr_ref_cnt->from_mem_pool());
  address_common_->pointer_ref_count_->set_original_ref_count(ptr_ref_cnt->original_ref_count());
  address_common_->pointer_ref_count_->set_ref_count(ptr_ref_cnt->ref_count());
  address_common_->pointer_ref_count_->set_dynamic_ref_count(ptr_ref_cnt->dynamic_ref_count());
  address_common_->pointer_ref_count_->set_deleter(ptr_ref_cnt->deleter());
  address_common_->pointer_ref_count_->set_is_ptr_persisted(ptr_ref_cnt->is_ptr_persisted());
  address_common_->pointer_ref_count_->set_new_ref_count(ptr_ref_cnt->new_ref_count());
}

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
  if (GetDevicePtr() != nullptr) {
    return true;
  }
  if (hete_info_ == nullptr) {
    return false;
  }
  return hete_info_->host_ptr_ != nullptr || !hete_info_->file_name_.empty();
}

bool DeviceAddress::IsNotNeedAlloc() const {
  return IsPtrValid() || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed);
}

bool DeviceAddress::IsNotNeedAllocWOLock() const {
  return (GetDevicePtr() != nullptr) || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed);
}

// Return the valid device ptr.
void *DeviceAddress::GetValidPtr(size_t) {
  if (user_data() == nullptr || (!need_sync_user_data_)) {
    return GetDevicePtr();
  }
  std::lock_guard<std::mutex> lock(ptr_mutex_);
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
  set_managed_by_somas(other->managed_by_somas());
  if (this->heterogeneous_info() != nullptr) {
    other->set_heterogeneous_info(std::make_shared<HeterogeneousInfo>());
    *(other->heterogeneous_info()) = *(this->heterogeneous_info());
    this->heterogeneous_info()->host_ptr_ = nullptr;
    this->heterogeneous_info()->file_name_ = "";
  }
}

const UserDataPtr &DeviceAddress::user_data() const { return user_data_; }

void DeviceAddress::set_user_data(const UserDataPtr &user_data) { user_data_ = user_data; }

const ShapeVector &DeviceAddress::host_shape() const { return host_shape_; }

void DeviceAddress::set_host_shape(const ShapeVector &host_shape) { host_shape_ = host_shape; }

HeterogeneousInfoPtr DeviceAddress::heterogeneous_info() const { return hete_info_; }

void DeviceAddress::set_heterogeneous_info(HeterogeneousInfoPtr hete_info) { hete_info_ = hete_info; }

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
void DeviceAddress::set_address_common(const AddressCommonPtr &address_common) { address_common_ = address_common; }

ContinuousDeviceAddressesPtr DeviceAddress::continuous_device_addresses() const { return continuous_device_addresses_; }

void DeviceAddress::set_continuous_device_addresses(const ContinuousDeviceAddressesPtr &continuous_device_addresses) {
  continuous_device_addresses_ = continuous_device_addresses;
}
}  // namespace device
}  // namespace mindspore
