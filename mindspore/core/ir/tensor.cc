/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ir/tensor.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <functional>
#include <memory>
#include <utility>
#include <algorithm>
#include <map>
#include <vector>
#include "mindapi/base/type_id.h"
#include "abstract/utils.h"
#include "abstract/abstract_value.h"
#include "base/complex_storage.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ordered_set.h"
#include "utils/system/env.h"
#include "utils/temp_file_manager.h"
#include "utils/ms_context.h"
#include "ir/device_address_maker.h"

namespace mindspore {
namespace tensor {
static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "T" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

static TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

std::unique_ptr<DeviceInfo> CopyDeviceInfo(const std::unique_ptr<DeviceInfo> &device_info) {
  return device_info == nullptr ? nullptr : std::make_unique<DeviceInfo>(device_info);
}

// Tensor chunk data.
template <typename T>
class TensorChunkData : public TensorDataImpl<T> {
 public:
  explicit TensorChunkData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~TensorChunkData() override = default;

  bool has_sub_data() const override { return true; }
};

// Tensor compression data.
template <typename T>
class CompressionTensorData : public TensorDataImpl<T> {
 public:
  explicit CompressionTensorData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~CompressionTensorData() override = default;
};

// TensorSubData is the base class to provide tensor data as a segment from an owner tensor data.
class TensorSubData : public TensorData {
 public:
  TensorSubData(const TensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : data_owner_(data_owner), data_offset_(offset), data_size_(data_size), ndim_(ndim) {}

  ~TensorSubData() override = default;

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  bool is_sub_data() const override { return true; }

  bool has_sub_data() const override { return false; }

  void *data() override {
    // Set data initialized if data() is called.
    data_initialized_ = true;
    auto start = static_cast<uint8_t *>(data_owner_->data().data());
    return static_cast<void *>(start + data_offset_);
  }

  const void *const_data() const override {
    if (!data_initialized_) {
      // Return nullptr if data not initialized.
      return nullptr;
    }
    auto start = static_cast<uint8_t *>(data_owner_->data().data());
    return static_cast<void *>(start + data_offset_);
  }

  // Get the owner Tensor.
  const TensorPtr &GetOwner() const { return data_owner_; }

  // Data offset in bytes.
  size_t data_offset() const { return data_offset_; }

 protected:
  const TensorPtr data_owner_;
  size_t data_offset_{0};
  size_t data_size_{0};
  size_t ndim_{0};
  bool data_initialized_{false};
};

// TensorSubDataImpl implements methods that rely on T.
template <typename T>
class TensorSubDataImpl : public TensorSubData {
 public:
  TensorSubDataImpl(const TensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : TensorSubData(data_owner, offset, data_size, ndim) {}

  ~TensorSubDataImpl() override = default;

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override {
    TensorStringifier<T> stringifier{static_cast<const T *>(const_data()), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }
};

TensorDataPtr MakeTensorSubData(const TensorPtr &owner, size_t offset, const TensorDataPtr &data) {
  if (data->nbytes() == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Tensor data size is 0.";
  }
  auto sub_data =
    tensor::MakeTensorData<TensorSubDataImpl>(owner->data_type(), owner, offset, data->size(), data->ndim());
  // If tensor data is initialized, copy it.
  if (data->const_data() != nullptr) {
    CopyTensorData(sub_data, data);
  }
  return sub_data;
}

// TensorChunk holds info for a chunk.
struct TensorChunk {
  size_t size{0};                  // chunk size in the number of elements.
  size_t bytes{0};                 // chunk size in bytes.
  std::vector<TensorPtr> tensors;  // tensors belong to this chunk.
};

static TypeId normalize_type(TypeId type_id) {
  if (type_id == kNumberTypeFloat) {
    // kNumberTypeFloat is an alias of kNumberTypeFloat32.
    return kNumberTypeFloat32;
  }
  return type_id;
}

Tensor::Tensor(const Tensor &tensor)
    : MetaTensor(tensor),
      contiguous_callback_(tensor.contiguous_callback_),
      id_(tensor.id_),
      tensor_name_(tensor.tensor_name_),
      version_(tensor.version_),
      device_sync_(tensor.device_sync_),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      device_info_(CopyDeviceInfo(tensor.device_info_)),
      pin_mem_register_(tensor.pin_mem_register_),
      sync_status_(tensor.sync_status_),
      compression_type_(tensor.compression_type_),
      is_forward_output_(tensor.is_forward_output_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      init_flag_(tensor.init_flag_),
      cache_enable_(tensor.cache_enable_),
      copy_done_flag_(tensor.copy_done_flag_) {
  user_data_ = tensor.user_data_;
}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      contiguous_callback_(tensor.contiguous_callback_),
      id_(tensor.data_type_ != data_type ? MakeId() : tensor.id_),
      tensor_name_(tensor.tensor_name_),
      version_(tensor.version_),
      device_sync_(tensor.device_sync_),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      device_info_(CopyDeviceInfo(tensor.device_info_)),
      pin_mem_register_(tensor.pin_mem_register_),
      sync_status_(tensor.sync_status_),
      compression_type_(tensor.compression_type_),
      is_forward_output_(tensor.is_forward_output_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      init_flag_(tensor.init_flag_),
      cache_enable_(tensor.cache_enable_),
      copy_done_flag_(tensor.copy_done_flag_) {
  // todo: tensor.astype
  MS_LOG(EXCEPTION) << "Not support change data type";
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (this == &tensor) {
    return *this;
  }
  is_forward_output_ = tensor.is_forward_output_;
  id_ = tensor.id_;
  sync_status_ = tensor.sync_status_;
  version_ = tensor.version_;
  device_sync_ = tensor.device_sync_;
  need_pipeline_sync_ = tensor.need_pipeline_sync_;
  lazy_callback_ = tensor.lazy_callback_;
  contiguous_callback_ = tensor.contiguous_callback_;
  user_data_ = tensor.user_data_;
  base_shape_ptr_ = tensor.base_shape_ptr_;
  auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
  init_flag_ = tensor.init_flag_;
  cache_enable_ = tensor.cache_enable_;
  cache_tensor_ptr_ = tensor.cache_tensor_ptr_;
  hashmap_tensor_ptr_ = tensor.hashmap_tensor_ptr_;
  pin_mem_register_ = tensor.pin_mem_register_;
  compression_type_ = tensor.compression_type_;
  tensor_name_ = tensor.tensor_name_;
  cast_dtype_ = tensor.cast_dtype_;
  graph_output_ = tensor.graph_output_;
  quant_params_ = tensor.quant_params_;
  updated_by_device_ = tensor.updated_by_device_;
  device_info_ = CopyDeviceInfo(tensor.device_info_);
  copy_done_flag_ = tensor.copy_done_flag_;
  return *this;
}

template <template <class> class ImplClass = TensorDataImpl, typename... Args>
TensorDataPtr MakeData(TypeId data_type, Args &&... args) {
  return MakeTensorData(data_type, args...);
}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, DeviceSyncPtr device_address)
    : MetaTensor(data_type, shape), id_(MakeId()), device_sync_(std::move(device_address)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape)
    : Tensor(data_type, shape, MakeDeviceAddress(data_type, shape)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : Tensor(data_type, shape, MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, data, data_len))) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type)
    : Tensor(data_type, shape,
             MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type))) {}

Tensor::Tensor(const std::vector<int64_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {static_cast<int>(input.size())}),
      id_(MakeId()),
      device_sync_(
        MakeDeviceAddress(data_type_, shape_, MakeTensorData(data_type_, shape_, input.data(), input.size()))) {}

Tensor::Tensor(const std::vector<int32_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      device_sync_(
        MakeDeviceAddress(data_type_, shape_, MakeTensorData(data_type_, shape_, input.data(), input.size()))) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      device_sync_(
        MakeDeviceAddress(data_type_, shape_, MakeTensorData(data_type_, shape_, input.data(), input.size()))) {}

Tensor::Tensor(const std::vector<float> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      device_sync_(
        MakeDeviceAddress(data_type_, shape_, MakeTensorData(data_type_, shape_, input.data(), input.size()))) {}

Tensor::Tensor(int64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(int32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(int16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt16), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(int8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt8), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(double input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(float input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(float16 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat16), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(float8_e5m2 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat8E5M2), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(float8_e4m3fn input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat8E4M3FN), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(hifloat8 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeHiFloat8), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

#ifndef KERNEL_EXECUTOR_ANDROID
Tensor::Tensor(bfloat16 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBFloat16), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}
#endif
Tensor::Tensor(uint64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt64), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(uint32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt32), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(uint16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt16), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(uint8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt8), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(bool input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBool), {}),
      id_(MakeId()),
      device_sync_(MakeDeviceAddress(data_type_, ShapeVector{}, MakeTensorData(data_type_, ShapeVector{}, input))) {}

Tensor::Tensor(TypeId data_type, size_t data_size)
    : Tensor(data_type, ShapeVector{static_cast<int64_t>(data_size)},
             MakeDeviceAddress(data_type, ShapeVector{static_cast<int64_t>(data_size)},
                               MakeTensorData<TensorChunkData>(data_type, data_size))) {}

Tensor::Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
               TensorCompressionType compression_type)
    : Tensor(origin_data_type, shape,
             MakeDeviceAddress(kNumberTypeInt8, shape,
                               MakeTensorData<CompressionTensorData>(kNumberTypeInt8, compression_data_size))) {
  compression_type_ = compression_type;
}

Tensor::~Tensor() {
  try {
    UnPinMemory();
    pin_mem_register_ = nullptr;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when destruct tensor. Error info " << e.what();
  }
}

bool Tensor::operator==(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && device_sync_ == tensor.device_sync_));
}

// Assign value to this tensor.
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    ExecuteLazyTask();
    contiguous_callback_ = tensor.contiguous_callback_;
    MetaTensor::operator=(tensor);
    device_sync_ = tensor.device_address();
    need_pipeline_sync_ = tensor.need_pipeline_sync_;
    is_forward_output_ = tensor.is_forward_output_;
    sync_status_ = tensor.sync_status_;
    version_ = tensor.version_;
    if (this->auto_grad_meta_data() != nullptr && this->auto_grad_meta_data()->input_type() == InputType::kInput) {
      MS_LOG(EXCEPTION)
        << "Can not modify tensor id of input tensor from network by assign value, this may caused by slice op, "
           "please check your code to avoid this error!";
    }
    if (!is_parameter_) {
      id_ = tensor.id_;
      auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
    }

    device_info_ = CopyDeviceInfo(tensor.device_info_);

    // Need execute callback when update host value of Tensor.
    ExecuteUpdateValueCallback();
  }
  return *this;
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  abstract::AbstractTensorPtr abs_tensor = nullptr;
  if (base_shape_ptr_ == nullptr) {
    auto tensor_shape = tens->shape();
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  } else {
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, base_shape_ptr_);
  }
  // if is parameter always no value.
  if (is_parameter_) {
    auto param_name = param_info_->name();
    auto ref_key = std::make_shared<RefKey>(param_name);
    abs_tensor = std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
  } else {
    abs_tensor->set_value(shared_from_base<Tensor>());
  }
  return abs_tensor;
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  MS_LOG(EXCEPTION) << "Not support!";
  // todo: tensor.cpu().equal(others.cpu())
  // return (&tensor == this || (MetaTensor::operator==(tensor) && data_->equals(*tensor.data_)));
}

TypeId Tensor::set_data_type(TypeId data_type) { MS_LOG(EXCEPTION) << "Not support set data_type!"; }

size_t Tensor::set_shape(const ShapeVector &shape) {
  MS_LOG(EXCEPTION) << "Not support change shape!";
  // todo: not support change shape.
  // if (DataSize() != SizeOf(shape)) {
  //   data_ = MakeTensorData(data_type_, shape);
  // }
  // return MetaTensor::set_shape(shape);
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToStringInternal(size_t limit_size) const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", value=";
  if (limit_size == 0 || DataSize() < limit_size) {
    // Only print data for small tensor.
    buf << ((DataDim() > 1) ? "\n" : "") << DataToString(false);
  } else {
    buf << "[...]";
  }
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string Tensor::ToString() const {
  constexpr size_t small_tensor_size = 30;
  return ToStringInternal(small_tensor_size);
}

std::string Tensor::ToStringNoLimit() const { return ToStringInternal(0); }

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", value=" << ((DataNDim() > 1) ? '\n' : ' ') << DataToString(true) << ')';
  return buf.str();
}

DeviceSyncPtr Tensor::device_address() const { return device_sync_; }

void Tensor::set_device_address(const DeviceSyncPtr &device_sync, bool need_update_ref_count) {
  device_sync_ = device_sync;
  // To support the old and new runtime coexistence, the output of old runtime may be the input of new runtime, so the
  // device address cannot be released through ref count and set max ref count in this scenario.
  if (need_update_ref_count && (device_sync_ != nullptr)) {
    device_sync_->set_original_ref_count(SIZE_MAX);
    device_sync_->ResetRefCount();
  }
}

const TensorStorageInfoPtr Tensor::storage_info() const {
  if (device_sync_ != nullptr) {
    return device_sync_->GetTensorStorageInfo();
  }
  return storage_info_;
}

void Tensor::set_storage_info(const TensorStorageInfoPtr &storage_info) { storage_info_ = storage_info; }

bool Tensor::is_contiguous() const {
  const auto &storage = storage_info();
  return storage == nullptr || storage->is_contiguous;
}

bool Tensor::NeedContiguous() const {
  const auto &storage = storage_info();
  if (storage == nullptr) {
    return false;
  }
  return !storage->is_contiguous || storage->storage_offset != 0;
}

std::vector<int64_t> Tensor::stride() const {
  const auto &storage = storage_info();
  if (storage != nullptr) {
    return storage->strides;
  }

  if (shape_.empty()) {
    return {};
  }
  std::vector<int64_t> ret(shape_.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape_.size() - 1; i > 0; --i) {
    stride *= shape_[i];
    ret[i - 1] = stride;
  }
  return ret;
}

const int64_t Tensor::storage_offset() const {
  const auto &storage = storage_info();
  return storage == nullptr ? 0 : SizeToLong(storage->storage_offset);
}

void Tensor::ExecuteLazyTask() const {
  if (lazy_callback_ != nullptr && (need_pipeline_sync_ || device_sync_ != nullptr)) {
    lazy_callback_();
  }
}

DeviceSyncPtr Tensor::CallContiguousCallback() const {
  DeviceSyncPtr contiguous_device_address = nullptr;
  if (contiguous_callback_ != nullptr && storage_info() != nullptr) {
    contiguous_device_address = contiguous_callback_(device_address());
    contiguous_device_address->set_original_ref_count(SIZE_MAX);
    contiguous_device_address->ResetRefCount();
  }
  return contiguous_device_address;
}

void Tensor::data_sync(bool need_wait, bool inpalce, bool sync_on_demand) const {
  if (need_wait) {
    ExecuteLazyTask();
  }

  if (device_sync_ == nullptr || device_sync_->GetMutablePtr() == nullptr) {
    return;
  }

  if (device_sync_->GetDeviceType() == device::DeviceType::kCPU) {
    MS_LOG(DEBUG) << "Already cpu device address. Skip data_sync.";
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), LongToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  auto contiguous_address = CallContiguousCallback();
  auto address = device_sync_;
  if (contiguous_address != nullptr) {
    address = contiguous_address;
    if (inpalce) {
      device_sync_ = contiguous_address;
    }
  }

  if (size != 0 && address->GetMutablePtr() != nullptr &&
      !address->SyncDeviceToHost(shape(), size, data_type(), data_c(), sync_on_demand)) {
    MS_LOG(INTERNAL_EXCEPTION) << "SyncDeviceToHost failed.";
  }

  // todo: support in device address
  // if (!data_->file_path().empty()) {
  //   device_sync_ = nullptr;
  // }
  sync_status_ = kNeedSyncHostToDevice;
}

TensorData &Tensor::data() {
  std::abort();
}

const TensorDataPtr &Tensor::data_ptr() const {
  std::abort();
}

const TensorData &Tensor::data() const {
  std::abort();
}

void Tensor::set_data(const TensorDataPtr &data) {
  std::abort();
}

void Tensor::ExecuteUpdateValueCallback() const {
  if (update_value_callback_ != nullptr) {
    update_value_callback_(this);
  }
}

void Tensor::SetDeviceInfo(const std::string &format, const TypePtr &data_type, const std::string &host_format) {
  DeviceInfo info(format, data_type, host_format);
  set_device_info(info);
}

void Tensor::data_sync_directly(const DeviceSync *const device_sync, bool need_wait) const {
  if (need_wait) {
    ExecuteLazyTask();
  }
  if (device_sync == nullptr) {
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), IntToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  auto contiguous_address = CallContiguousCallback();
  if (contiguous_address == nullptr) {
    if (size != 0 && !device_sync->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
      MS_LOG(INTERNAL_EXCEPTION) << "SyncDeviceToHost failed.";
    }
  } else {
    if (size != 0 && !contiguous_address->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
      MS_LOG(INTERNAL_EXCEPTION) << "SyncDeviceToHost failed.";
    }
  }

  sync_status_ = kNeedSyncHostToDevice;
}

bool Tensor::Offload(const std::string &file_path) {
  // todo: support in device address.
  //  if (file_path.empty()) {
  //    return false;
  //  }
  //
  //  auto fs = mindspore::system::Env::GetFileSystem();
  //  MS_EXCEPTION_IF_NULL(fs);
  //  MS_EXCEPTION_IF_NULL(data_);
  //  auto data_ptr = data_->data();
  //  auto file = fs->CreateWriteFile(file_path);
  //  MS_EXCEPTION_IF_NULL(file);
  //  TempFileManager::GetInstance().Register(file_path);
  //  bool success = file->PWrite(data_ptr, LongToSize(data_->nbytes()), 0);
  //  if (!file->Close()) {
  //    MS_LOG(WARNING) << "Close tensor file: " << file_path << " failed!";
  //  }
  //  if (!success) {
  //    MS_LOG(WARNING) << "Tensor write data to file: " << file_path << " failed!";
  //    return false;
  //  }
  //
  //  if (file_path == GetOffloadFilePath()) {
  //    data_->set_file_path("");
  //  }
  //
  //  data_ = tensor::MakeTensorData(data_type_, shape_);
  //  MS_EXCEPTION_IF_NULL(data_);
  //  data_->set_file_path(file_path);
  return true;
}

const std::string Tensor::GetOffloadFilePath() const {
  // todo
  //  if (data_ == nullptr) {
  //    return "";
  //  }
  //  return data_->file_path();
  return "";
}

std::pair<void *, size_t> Tensor::GetChunkOffset() const {
  // Get sub-data.
  auto sub_data = std::dynamic_pointer_cast<TensorSubData>(data_ptr());
  if (sub_data == nullptr) {
    return {nullptr, 0};
  }
  // Get owner tensor from sub-data.
  auto owner_tensor = sub_data->GetOwner();
  MS_EXCEPTION_IF_NULL(owner_tensor);
  return {owner_tensor->data_c(), sub_data->data_offset()};
}

static std::map<TypeId, std::vector<TensorChunk>> GroupingTensors(const TensorPtrList &tensors, size_t fusion_size) {
  // Use std::map to keep order by type id.
  std::map<TypeId, std::vector<TensorChunk>> group_info;
  for (auto &tensor : tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_bytes = static_cast<size_t>(tensor->DataNBytes());
    if ((fusion_size != 0) && (tensor_bytes > fusion_size)) {
      MS_LOG(EXCEPTION) << "Fusion size " << fusion_size << " is too small for a tensor size " << tensor_bytes << ".";
    }
    auto &chunks = group_info[normalize_type(tensor->data_type())];
    if (chunks.empty()) {
      (void)chunks.emplace_back();
    }
    if ((fusion_size != 0) && (chunks.back().bytes + tensor_bytes > fusion_size)) {
      (void)chunks.emplace_back();
    }
    auto &chunk = chunks.back();
    chunk.size += tensor->DataSize();
    chunk.bytes += tensor_bytes;
    (void)chunk.tensors.emplace_back(tensor);
  }
  return group_info;
}

TensorPtrList Tensor::FlattenTensors(const TensorPtrList &tensors, size_t fusion_size) {
  // Result tensor list.
  TensorPtrList result_list;
  // Grouping tensors by data type and fusion size.
  auto group_info = GroupingTensors(tensors, fusion_size);
  // Create chunk tensors and copy data to them.
  for (auto &type_group : group_info) {
    auto chunk_dtype = normalize_type(type_group.first);
    for (auto &chunk : type_group.second) {
      // Create chunk thensor as a lazy initialized tensor, the tensor data
      // will be allocated when we begin to copy small tensors data into it.
      auto chunk_tensor = std::make_shared<Tensor>(chunk_dtype, chunk.size);
      // Reset and copy tensors data.
      size_t offset = 0;
      for (auto &tensor : chunk.tensors) {
        auto sub_data = MakeTensorSubData(chunk_tensor, offset, tensor->data_ptr());
        offset += static_cast<size_t>(sub_data->nbytes());
        tensor->set_data(sub_data);
      }
      // Save chunk tensor to result list.
      (void)result_list.emplace_back(std::move(chunk_tensor));
    }
  }
  return result_list;
}

bool Tensor::IsFlattened(const TensorPtrList &tensors) {
  // Tensor data is flattened if all tensors data are TensorSubData.
  return std::all_of(tensors.begin(), tensors.end(), [](const TensorPtr &tensor) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto data_ptr = tensor->data_ptr().get();
    return dynamic_cast<TensorSubData *>(data_ptr) != nullptr;
  });
}

const TensorPtr Tensor::GetFlattenedTensor(const TensorPtr &tensor) {
  // Get sub-data.
  auto sub_data = std::dynamic_pointer_cast<TensorSubData>(tensor->data_ptr());
  if (sub_data == nullptr) {
    MS_LOG(WARNING) << "Tensors are not flattened.";
    return nullptr;
  }
  // Get owner tensor from sub-data.
  auto owner_tensor = std::dynamic_pointer_cast<Tensor>(sub_data->GetOwner());
  MS_EXCEPTION_IF_NULL(owner_tensor);
  return owner_tensor;
}

TensorPtrList Tensor::GetFlattenedTensors(const TensorPtrList &tensors) {
  // Use std::map to keep order by type id.
  std::map<TypeId, OrderedSet<TensorPtr>> chunk_map;
  for (auto &tensor : tensors) {
    auto owner_tensor = GetFlattenedTensor(tensor);
    if (owner_tensor == nullptr) {
      return {};
    }
    // Add as chunk tensor by its data type.
    auto chunk_dtype = normalize_type(tensor->data_type());
    chunk_map[chunk_dtype].add(owner_tensor);
  }
  // Generate result tensor list.
  TensorPtrList result_tensors;
  for (auto &entry : chunk_map) {
    auto &chunk_tensors = entry.second;
    (void)result_tensors.insert(result_tensors.end(), chunk_tensors.begin(), chunk_tensors.end());
  }
  return result_tensors;
}

bool Tensor::CheckStub() {
#if defined(WITH_BACKEND)
  return false;
#else
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string backend_name = context_ptr->backend_policy();
  if (backend_name == "vm") {
    return false;
  }
  return true;
#endif
}

size_t Tensor::GetFusionSize(const TensorPtrList &flat_tensors) {
  size_t fusion_size = 0;
  std::map<TypeId, size_t> type_groups;
  for (auto &tensor : flat_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_bytes = static_cast<size_t>(tensor->DataNBytes());
    if (tensor_bytes > fusion_size) {
      fusion_size = tensor_bytes;
    }
    ++type_groups[tensor->data_type()];
  }
  const bool only_one_chunk_for_each_type =
    std::all_of(type_groups.begin(), type_groups.end(), [](auto const &e) { return e.second == 1; });
  if (only_one_chunk_for_each_type) {
    return 0;
  }
  return fusion_size;
}

bool Tensor::is_persistent_data() const { return this->data().is_persistent_data(); }

void Tensor::PinMemory(PinnedMemRegister *pin_mem_register) {
  if (pin_mem_register == nullptr) {
    return;
  }
  pin_mem_register_ = pin_mem_register;
  pin_mem_register_->RegisterPinnedMem(data_c(), Size());
}

void Tensor::UnPinMemory() {
  if (pin_mem_register_ == nullptr) {
    return;
  }
  pin_mem_register_->UnRegisterPinnedMem(data_c());
}

CSRTensor::CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indptr_(indptr), indices_(indices), values_(values) {}

std::string CSRTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indptr_);
  auto dtype = values_->Dtype();
  values_->data_sync(true);
  indices_->data_sync(true);
  indptr_->data_sync(true);
  buf << "CSRTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", indptr=";
  buf << indptr_->ToString() << ", indices=" << indices_->ToString() << ", values=";
  buf << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr CSRTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }

  auto indptr = indptr_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indptr, indices, values, shape};

  return std::make_shared<abstract::AbstractCSRTensor>(element_list);
}

const size_t CSRTensor::GetSizeAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_->DataNBytes();
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_->DataNBytes();
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_->DataNBytes();
  } else if (index >= kIndicesIdx && index < kShapeIdx + shape().size()) {
    return sizeof(int64_t);
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

TensorPtr CSRTensor::GetTensorAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_;
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return std::make_shared<tensor::Tensor>(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

TensorPtr COOTensor::GetTensorAt(size_t index) const {
  if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return std::make_shared<tensor::Tensor>(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for COOTensor: " << ToString();
}

std::string COOTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  indices_->data_sync(true);
  values_->data_sync(true);
  auto dtype = values_->Dtype();
  buf << "COOTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr COOTensor::ToAbstract() {
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indices, values, shape};

  return std::make_shared<abstract::AbstractCOOTensor>(element_list);
}

std::string RowTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  buf << "RowTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr RowTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto abs_sparse_tensor = std::make_shared<abstract::AbstractRowTensor>(dtype, shape_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  abs_sparse_tensor->set_indices(indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_sparse_tensor->set_values(values_->ToAbstract()->cast<abstract::AbstractTensorPtr>());

  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  abs_sparse_tensor->set_dense_shape(std::make_shared<abstract::AbstractTuple>(abstract_shape));

  return abs_sparse_tensor;
}

std::string ShapeToString(const ShapeVector &shape) {
  std::string str = "[";
  const size_t count = shape.size();
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      str.append(", ");
    }
    str.append(std::to_string(shape[i]));
  }
  return str.append("]");
}
}  // namespace tensor
}  // namespace mindspore
