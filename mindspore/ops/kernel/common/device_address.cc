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
DevicePtrDeleterMakerFunc g_deleter_func[static_cast<int>(device::DeviceType::kDeviceEnd)];
void SetDevicePtrDeleterMaker(device::DeviceType device_type, DevicePtrDeleterMakerFunc &&func) {
  MS_LOG(DEBUG) << "Resigter device ptr deleter function for device type:" << device::GetDeviceNameByType(device_type);
  g_deleter_func[static_cast<int>(device_type)] = func;
}

using ContinuousDeviceAddressesPtr = std::shared_ptr<std::vector<std::weak_ptr<DeviceAddress>>>;

DeviceAddress::DeviceAddress() { pointer_ref_count_ = std::make_shared<PointerRefCount>(); }

DeviceAddress::DeviceAddress(void *device_ptr, size_t size)
    : pointer_ref_count_(std::make_shared<PointerRefCount>(device_ptr)), size_(size) {}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &device_name)
    : pointer_ref_count_(std::make_shared<PointerRefCount>(ptr)), size_(size) {
  device_type_ = device::GetDeviceTypeByName(device_name);
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id,
                             const std::string &device_name) {
  pointer_ref_count_ = std::make_shared<PointerRefCount>();
  pointer_ref_count_->set_ptr(ptr);
  size_ = size;
  dtype_id_ = type_id;
  device_type_ = device::GetDeviceTypeByName(device_name);
  format_ = kernel::GetFormatFromStrToEnum(format);
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id) {
  pointer_ref_count_ = std::make_shared<PointerRefCount>();
  pointer_ref_count_->set_ptr(ptr);
  size_ = size;
  device_type_ = device::GetDeviceTypeByName(device_name);
  device_id_ = device_id;
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id,
                             const std::string &device_name, uint32_t device_id) {
  pointer_ref_count_ = std::make_shared<PointerRefCount>();
  pointer_ref_count_->set_ptr(ptr);
  size_ = size;
  device_type_ = device::GetDeviceTypeByName(device_name);
  dtype_id_ = type_id;
  format_ = kernel::GetFormatFromStrToEnum(format);
  device_id_ = device_id;
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                             TypeId type_id, const std::string &device_name, uint32_t device_id, uint32_t stream_id)
    : pointer_ref_count_(std::make_shared<PointerRefCount>(ptr)),
      stream_id_(stream_id),
      size_(size),
      format_(format),
      dtype_id_(type_id),
      device_type_(device::GetDeviceTypeByName(device_name)),
      device_id_(device_id),
      shape_vector_(shape_vector) {
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                             const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
    : node_index_(node_index) {
  pointer_ref_count_ = std::make_shared<PointerRefCount>();
  pointer_ref_count_->set_ptr(ptr);
  size_ = size;
  device_type_ = device::GetDeviceTypeByName(device_name);
  dtype_id_ = type_id;
  format_ = kernel::GetFormatFromStrToEnum(format);
  device_id_ = device_id;
  SetDevicePtrDeleter();
}

DeviceAddress::DeviceAddress(const DeviceAddress &other) {
  pointer_ref_count_ =
    other.pointer_ref_count_ != nullptr
      ? std::make_shared<PointerRefCount>(other.pointer_ref_count_->ptr(), other.pointer_ref_count_->deleter(),
                                          other.pointer_ref_count_->allocator())
      : std::make_shared<PointerRefCount>();
  tensor_storage_info_ = other.tensor_storage_info_;
  stream_id_ = other.stream_id_;
  size_ = other.size_;
  format_ = other.format_;
  dtype_id_ = other.dtype_id_;
  device_id_ = other.device_id_;
  device_type_ = other.device_type_;
  dtype_id_ = other.dtype_id_;
  shape_vector_ = other.shape_vector_;
  padding_type_ = other.padding_type();
  is_view_ = other.is_view();
  deleter_ = other.deleter();
  host_shape_ = other.host_shape();
  SetDevicePtrDeleter();
}

DeviceAddress::~DeviceAddress() {
  if (IS_OUTPUT_ON(mindspore::kDebug) && pointer_ref_count_ != nullptr &&
      pointer_ref_count_->new_ref_count() != SIZE_MAX && GetPtr() != nullptr) {
    MS_LOG(DEBUG) << "Maybe memory leak detect in device address:" << ToString();
  }
  if (!from_mem_pool() && deleter_ && GetDevicePtr() != nullptr) {
    deleter_(static_cast<uint8_t *>(GetDevicePtr()));
    SetDevicePtr(nullptr);
  } else {
    pointer_ref_count_ = nullptr;
  }
}

std::string DeviceAddress::ToString() const {
  std::ostringstream ofs;
  ofs << this << " device type:" << GetDeviceType() << " host shape:" << host_shape_
      << " tensor storage info:" << tensor_storage_info_;
  if (tensor_storage_info_ != nullptr) {
    ofs << tensor_storage_info_->ToString();
  }
  ofs << " size:" << size_ << " format:" << format_ << " dtype:" << dtype_id_ << " device id:" << device_id_
      << " device name:" << device::GetDeviceNameByType(device_type_) << " shape vector:{";
  std::for_each(shape_vector_.begin(), shape_vector_.end(), [&ofs](ShapeValueDType axis) { ofs << axis << " "; });
  ofs << "} point ref count:";
  if (pointer_ref_count_ == nullptr) {
    ofs << "0";
  } else {
    ofs << pointer_ref_count_->ToString();
  }
  if (hete_info_ != nullptr) {
    ofs << " hete info:" << hete_info_->ToString();
  }
  const auto &node_index = GetNodeIndex();
  if (node_index.first != nullptr) {
    ofs << " node:" << node_index.first->fullname_with_scope() << " index:" << node_index.second;
  }
  ofs << " device address deleter:" << (deleter_ != nullptr) << " is view:" << is_view_
      << " from persist mem:" << from_persistent_mem_ << " need recycle:" << need_recycle_
      << " padding type:" << padding_type_ << " status:" << status_;
  return ofs.str();
}

const void *DeviceAddress::GetPtr() const { return GetDevicePtr(); }

void DeviceAddress::set_ptr(void *ptr) {
  pointer_ref_count_->set_ptr(ptr);
  if (ptr != nullptr) {
    status_ = DeviceAddressStatus::kInDevice;
  }
}

size_t DeviceAddress::GetSize() const {
  if (tensor_storage_info_ && (tensor_storage_info_->ori_size != 0)) {
    return tensor_storage_info_->ori_size;
  }
  return size();
}

void DeviceAddress::SetSize(size_t size) { size_ = size; }

std::string DeviceAddress::format() const { return kernel::GetFormatFromEnumToStr(format_); }

void DeviceAddress::set_format(const std::string &format) { format_ = kernel::GetFormatFromStrToEnum(format); }

const std::string &DeviceAddress::padding_type() const { return padding_type_; }

void DeviceAddress::set_padding_type(const std::string &padding_type) { padding_type_ = padding_type; }

TypeId DeviceAddress::type_id() const { return dtype_id_; }

void DeviceAddress::set_type_id(TypeId dtype_id) { dtype_id_ = dtype_id; }

bool DeviceAddress::from_mem_pool() const { return pointer_ref_count_->from_mem_pool(); }

void DeviceAddress::set_from_mem_pool(bool from_mem_pool) const {
  pointer_ref_count_->set_from_mem_pool(from_mem_pool);
}

void DeviceAddress::set_communication_ptr(uint8_t *communication_ptr) { MS_LOG(EXCEPTION) << "Not implemented error."; }

bool DeviceAddress::is_ptr_persisted() const { return pointer_ref_count_->is_ptr_persisted(); }

void DeviceAddress::set_is_ptr_persisted(bool is_ptr_persisted) {
  pointer_ref_count_->set_is_ptr_persisted(is_ptr_persisted);
}

bool DeviceAddress::from_persistent_mem() const { return from_persistent_mem_; }

void DeviceAddress::set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }

bool DeviceAddress::need_recycle() const { return need_recycle_; }

void DeviceAddress::set_need_recycle(bool need_recycle) { need_recycle_ = need_recycle; }

void DeviceAddress::set_status(DeviceAddressStatus status) { status_ = status; }

DeviceAddressStatus DeviceAddress::status() const { return status_; }

void *DeviceAddress::GetMutablePtr() const { return GetDevicePtr(); }

const ShapeVector &DeviceAddress::GetShapeVector() const { return shape_vector_; }

void DeviceAddress::SetShapeVector(const ShapeVector &shape_vector) { shape_vector_ = shape_vector; }

TensorStorageInfoPtr DeviceAddress::GetTensorStorageInfo() const { return tensor_storage_info_; }

void DeviceAddress::set_tensor_storage_info(const TensorStorageInfoPtr &tensor_storage_info) {
  tensor_storage_info_ = tensor_storage_info;
}

device::DeviceType DeviceAddress::GetDeviceType() const { return device_type_; }
void DeviceAddress::SetDeviceType(const device::DeviceType &device_type) { device_type_ = device_type; }

uint32_t DeviceAddress::device_id() const { return device_id_; }
void DeviceAddress::set_device_id(uint32_t device_id) { device_id_ = device_id; }

void DeviceAddress::set_stream_id(uint32_t stream_id) { stream_id_ = stream_id; }

const uint32_t DeviceAddress::stream_id() const { return stream_id_; }

bool DeviceAddress::managed_by_somas() const { return managed_by_somas_; }

void DeviceAddress::set_managed_by_somas(bool managed_by_somas) { managed_by_somas_ = managed_by_somas; }

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
  pointer_ref_count_->IncreaseNewRefCount(i);
  MS_LOG(DEBUG) << "Op:" << op_name << " increase new ref count for device address:" << ToString();
}

void DeviceAddress::IncreaseNewRefCount(size_t i) { pointer_ref_count_->IncreaseNewRefCount(i); }

size_t DeviceAddress::DecreaseNewRefCount(const std::string &op_name) {
  size_t ref_count = pointer_ref_count_->DecreaseNewRefCount();
  MS_LOG(DEBUG) << "Op:" << op_name << " decrease new ref count for device address:" << ToString();
  return ref_count;
}

void DeviceAddress::set_new_ref_count(size_t new_ref_count) const {
  pointer_ref_count_->set_new_ref_count(new_ref_count);
}

size_t DeviceAddress::new_ref_count() const { return pointer_ref_count_->new_ref_count(); }

void DeviceAddress::set_original_ref_count(size_t original_ref_count) const {
  pointer_ref_count_->set_original_ref_count(original_ref_count);
}

size_t DeviceAddress::original_ref_count() const { return pointer_ref_count_->original_ref_count(); }

void DeviceAddress::set_ref_count(size_t ref_count) const { pointer_ref_count_->set_ref_count(ref_count); }

size_t DeviceAddress::ref_count() const { return pointer_ref_count_->ref_count(); }

void DeviceAddress::ResetRefCount() { pointer_ref_count_->ResetRefCount(); }

void DeviceAddress::IncreaseOriginalRefCount() {
  if (original_ref_count() < SIZE_MAX) {
    pointer_ref_count_->IncreaseOriginalRefCount();
  }
}

void DeviceAddress::DecreaseOriginalRefCount() {
  if ((original_ref_count() < SIZE_MAX) && (original_ref_count() > 0)) {
    pointer_ref_count_->DecreaseOriginalRefCount();
  }
}

void DeviceAddress::IncreaseRefCount(size_t increase_cnt) { pointer_ref_count_->IncreaseRefCount(increase_cnt); }

size_t DeviceAddress::DecreaseRefCount() { return pointer_ref_count_->DecreaseRefCount(); }

void DeviceAddress::set_dynamic_ref_count(int32_t dynamic_ref_count) {
  pointer_ref_count_->set_dynamic_ref_count(dynamic_ref_count);
}

int32_t DeviceAddress::dynamic_ref_count() const { return pointer_ref_count_->dynamic_ref_count(); }

void DeviceAddress::IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt) {
  pointer_ref_count_->IncreaseDynamicRefCount(op_object, increase_cnt);
}

void DeviceAddress::IncreaseDynamicRefCount(const std::string &op_object) {
  pointer_ref_count_->IncreaseDynamicRefCount(op_object);
}

int32_t DeviceAddress::DecreaseDynamicRefCount(const std::string &op_object) {
  return pointer_ref_count_->DecreaseDynamicRefCount(op_object);
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

void DeviceAddress::Swap(DeviceAddress *other) {
  MS_EXCEPTION_IF_NULL(other);
  if (other == this) {
    return;
  }
  other->SetDevicePtr(GetDevicePtr());

  other->set_from_mem_pool(this->from_mem_pool());
  other->set_deleter(deleter());
  SetDevicePtr(nullptr);
  this->set_from_mem_pool(false);
  deleter_ = nullptr;
  set_managed_by_somas(other->managed_by_somas());
}

const ShapeVector &DeviceAddress::host_shape() const { return host_shape_; }

void DeviceAddress::set_host_shape(const ShapeVector &host_shape) { host_shape_ = host_shape; }

HeterogeneousInfoPtr DeviceAddress::heterogeneous_info() const { return hete_info_; }

void DeviceAddress::set_heterogeneous_info(HeterogeneousInfoPtr hete_info) { hete_info_ = hete_info; }

std::pair<AnfNodeWeakPtr, size_t> DeviceAddress::node_index() const { return node_index_; }

void DeviceAddress::set_deleter(const std::function<void(uint8_t *)> &deleter) { deleter_ = deleter; }

void DeviceAddress::SetPointerRefCountDeleter(std::function<void(void *, bool)> &&deleter) {
  pointer_ref_count()->set_deleter(deleter);
}

std::function<void(uint8_t *)> DeviceAddress::deleter() const { return deleter_; }

const PointerRefCountPtr &DeviceAddress::pointer_ref_count() const { return pointer_ref_count_; }

void DeviceAddress::set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt) {
  MS_EXCEPTION_IF_NULL(ptr_ref_cnt);
  pointer_ref_count_ = ptr_ref_cnt;
}

void DeviceAddress::set_is_view(bool is_view) { is_view_ = is_view; }

bool DeviceAddress::is_view() const { return is_view_; }

DeviceAddressPtr DeviceAddress::CloneDeviceAddress() { return std::make_shared<DeviceAddress>(*this); }

void DeviceAddress::set_data(tensor::TensorDataPtr &&data) {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    data_ = std::move(data);
  } else {
    MS_LOG(DEBUG) << "Skip device address set_data";
  }
}

const tensor::TensorDataPtr &DeviceAddress::data() const {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    return data_;
  } else {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }
}

bool DeviceAddress::has_data() const {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    return data_ != nullptr;
  } else {
    return false;
  }
}

namespace {
DevicePtrDeleterMakerFunc GetDevicePtrDeleterMaker(device::DeviceType device_type) {
  auto maker = g_deleter_func[static_cast<int>(device_type)];
  return maker;
}
}  // namespace

void DeviceAddress::SetDevicePtrDeleter() {
  if (pointer_ref_count_ == nullptr) {
    return;
  }
  auto deleter = GetDevicePtrDeleterMaker(GetDeviceType());
  if (deleter != nullptr) {
    pointer_ref_count_->set_deleter(deleter);
  } else {
    MS_LOG(INFO) << "Get device ptr deleter function failed, device type: "
                 << device::GetDeviceNameByType(GetDeviceType());
  }
}

void DeviceAddress::ClearDeviceMemory() {
  if (pointer_ref_count_ == nullptr) {
    return;
  }
  auto deleter = pointer_ref_count_->deleter();
  if (GetDevicePtr() != nullptr && from_mem_pool() && deleter) {
    deleter(GetDevicePtr(), from_mem_pool());
    SetDevicePtr(nullptr);
  }
}
}  // namespace device
}  // namespace mindspore
