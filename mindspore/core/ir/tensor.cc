/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <utility>
#include <algorithm>
#include <map>
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
  MS_LOG(ERROR) << "Not support change tensor data type";
  std::abort();
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

Tensor::Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
               TensorCompressionType compression_type)
    : Tensor(
        origin_data_type, shape,
        MakeDeviceAddress(kNumberTypeInt8, shape,
                          MakeTensorData(kNumberTypeInt8, ShapeVector{static_cast<int64_t>(compression_data_size)}))) {
  compression_type_ = compression_type;
}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, bool ref_mem, void *data)
    : Tensor(data_type, shape, MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, ref_mem, data))) {}

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

// assign value to this tensor
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

bool TensorEqual(const Tensor &self, const Tensor &other) {
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto self_ptr = static_cast<const uint8_t *>(self_cpu->data_c());
  auto other_ptr = static_cast<const uint8_t *>(other_cpu->data_c());
  if (self_ptr == nullptr || other_ptr == nullptr) {
    return false;
  }
  if (self_ptr == other_ptr) {
    return true;
  }
  return self.DataNDim() == other.DataNDim() && self.DataNBytes() == other.DataNBytes() &&
         std::equal(self_ptr, self_ptr + self.DataNBytes(), other_ptr);
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  return (&tensor == this || (MetaTensor::operator==(tensor) && TensorEqual(*this, tensor)));
}

TypeId Tensor::set_data_type(TypeId data_type) {
  if (data_type != data_type_) {
    MS_EXCEPTION_IF_NULL(device_sync_);
    if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
      auto cpu_tensor = cpu();
      device_sync_ = cpu_tensor->device_address();
    }
    auto new_dtype_address = MakeDeviceAddress(data_type, shape_, true);
    MS_EXCEPTION_IF_NULL(new_dtype_address);
    if (!SyncCopy(new_dtype_address, device_sync_, device_sync_->stream_id())) {
      MS_LOG(EXCEPTION) << "Sync copy failed";
    }
    device_sync_ = new_dtype_address;
    id_ = MakeId();
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}

size_t Tensor::set_shape(const ShapeVector &shape) {
  if (DataSize() < SizeOf(shape)) {
    MS_LOG(WARNING) << "It's invalid to set " << ToString() << " shape to " << shape;
  }
  return MetaTensor::set_shape(shape);
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

void *Tensor::data_c() const {
  if (device_sync_ == nullptr) {
    MS_LOG(ERROR) << "Cannot access uninitialized tensor data";
    std::abort();
  }
  if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
    MS_LOG(ERROR) << "Only cpu Tensor can access data.";
    std::abort();
  }
  return device_sync_->GetMutablePtr();
}

TensorPtr Tensor::cpu() const {
  // todo: check stream id!!!
  ExecuteLazyTask();
  DeviceSyncPtr device_address;
  auto contiguous_address = CallContiguousCallback();
  if (contiguous_address != nullptr) {
    device_address = contiguous_address;
  } else {
    device_address = device_sync_;
  }
  if (device_address == nullptr) {
    MS_LOG(ERROR) << "Can't do cpu() for uninitialized tensor";
    return std::make_shared<Tensor>(data_type_, shape_, device_address);
  }
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    return std::make_shared<Tensor>(data_type_, shape_, device_address);
  }
  auto dst = MakeDeviceAddress(data_type_, shape_, true);
  MS_EXCEPTION_IF_NULL(dst);
  SyncCopy(dst, device_address, device_address->stream_id());
  return std::make_shared<Tensor>(data_type_, shape_, dst);
}

bool Tensor::to_device() {
  if (to_device_callback_ == nullptr) {
    MS_LOG(DEBUG) << "No callback found.";
    return true;
  }
  MS_LOG(DEBUG) << "Run callback.";
  bool ret = to_device_callback_();
  to_device_callback_ = nullptr;
  return ret;
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
}

std::string Tensor::DataToString(bool use_comma) const {
  if (device_sync_ == nullptr) {
    return "<uninitialized>";
  }
  if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
    return "<" + device::GetDeviceNameByType(device_sync_->GetDeviceType()) + ">";
  }
  return GetTensorDataString(data_type_, shape_, device_sync_->GetMutablePtr(), DataSize(), DataDim(), use_comma);
}

void *Tensor::unsafe_data() {
  if (device_sync_ == nullptr) {
    return nullptr;
  }
  return device_sync_->GetMutablePtr();
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

void Tensor::data_sync_directly(const DeviceSync *const device_sync, bool need_wait) const {}

bool Tensor::Offload(const std::string &file_path) {
  if (file_path.empty()) {
    return false;
  }

  auto fs = mindspore::system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(file_path);
  MS_EXCEPTION_IF_NULL(file);
  TempFileManager::GetInstance().Register(file_path);
  bool success = file->PWrite(data_c(), DataNBytes(), 0);
  if (!file->Close()) {
    MS_LOG(WARNING) << "Close tensor file: " << file_path << " failed!";
  }
  if (!success) {
    MS_LOG(WARNING) << "Tensor write data to file: " << file_path << " failed!";
    return false;
  }

  if (file_path == GetOffloadFilePath()) {
    offload_file_.clear();
  }

  device_sync_ = MakeDeviceAddress(data_type_, shape_);
  offload_file_ = file_path;
  return true;
}

const std::string Tensor::GetOffloadFilePath() const { return offload_file_; }

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
  auto values_cpu = values_->cpu();
  auto indices_cpu = indices_->cpu();
  auto indptr_cpu = indptr_->cpu();
  buf << "CSRTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", indptr=";
  buf << indptr_cpu->ToString() << ", indices=" << indices_cpu->ToString() << ", values=";
  buf << values_cpu->ToString() << ")";
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
  auto indices_cpu = indices_->cpu();
  auto values_cpu = values_->cpu();
  auto dtype = values_->Dtype();
  buf << "COOTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_cpu->ToString() << ", values=" << values_cpu->ToString() << ")";
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
