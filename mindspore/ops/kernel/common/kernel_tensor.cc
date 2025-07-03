/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "common/kernel_tensor.h"
#include "common/format_utils.h"
#include "common/utils/utils.h"
#include "common/kernel_callback.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"

namespace mindspore::kernel {
namespace {
void TransposeDefaultShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);
  *device_shape_vector = *host_shape_vector;
}

void TransposeNCHWShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);
  if (host_shape_vector->size() != kDim4) {
    MS_LOG(EXCEPTION) << "The host shape dims should be 4, but got: " << host_shape_vector->size();
  }
  *device_shape_vector = *host_shape_vector;
}

void TransposeNHWCShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);

  if (host_shape_vector->size() != kDim4) {
    MS_LOG(EXCEPTION) << "The host shape dims should be 4, but got: " << host_shape_vector->size();
  }
  device_shape_vector->resize(kDim4);

  device_shape_vector->at(kIndex0) = host_shape_vector->at(kIndex0);
  device_shape_vector->at(kIndex1) = host_shape_vector->at(kIndex2);
  device_shape_vector->at(kIndex2) = host_shape_vector->at(kIndex3);
  device_shape_vector->at(kIndex3) = host_shape_vector->at(kIndex1);
}

ShapeVector GetShapeVectorByBaseShape(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::NoShape>()) {
    return {};
  } else if (base_shape->isa<abstract::Shape>()) {
    return base_shape->cast<abstract::ShapePtr>()->shape();
  } else if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    return {-1};
  } else if (base_shape->isa<abstract::SequenceShape>()) {
    const auto &sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    if (sequence_shape->size() == 0) {
      return {0};
    }
    ShapeVector shape_vector = {SizeToLong(sequence_shape->size())};
    const auto &sub_shape_vector = GetShapeVectorByBaseShape(sequence_shape->shape()[0]);
    shape_vector.insert(shape_vector.end(), sub_shape_vector.begin(), sub_shape_vector.end());
    return shape_vector;
  }
  MS_LOG(EXCEPTION) << "Invalid shape:" << base_shape->ToString();
}
}  // namespace

KernelHostInfo::KernelHostInfo(const KernelHostInfo &other) {
  shape_vector_after_format_trasform_ = other.shape_vector_after_format_trasform_;
  type_id_ = other.type_id_;
  kernel_tensor_value_ = other.kernel_tensor_value_;
}

KernelTensor::KernelTensor() { address_common_ = std::make_shared<AddressCommon>(); }

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  host_info_ = std::make_unique<KernelHostInfo>();
  address_common_ = std::make_shared<AddressCommon>();

  if (type) {
    SetType(type);
  }
  if (shape) {
    // Note: for performance, the function `SetShape` uses host_info_->type_id_, so need to SetType first.
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }
}

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, TypeId dtype_id, const ShapeVector &host_shape) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address_ = device_address;
  address_common_ = device_address_->address_common();
  device_address_->set_host_shape(host_shape);
  if (dtype_id == kTypeUnknown) {
    SetType(TypeIdToType(dtype_id));
  } else {
    SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  }
}

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, const abstract::BaseShapePtr &shape,
                           const TypePtr &type, const ValuePtr &value, void *device_ptr, size_t size,
                           const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
                           const string &device_name, uint32_t device_id)
    : KernelTensor(shape, type, value) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address_ = device_address;
  address_common_->pointer_ref_count_->set_ptr(device_ptr);
  auto pointer_ref_count = device_address_->address_common()->pointer_ref_count_;
  address_common_->pointer_ref_count_->set_deleter(pointer_ref_count->deleter());
  address_common_->size_ = size;
  address_common_->format_ = GetFormatFromStrToEnum(format);
  address_common_->dtype_id_ = dtype_id;
  address_common_->device_name_ = device_name;
  address_common_->device_id_ = device_id;
  device_address_->set_address_common(address_common_);
  device_address_->set_host_shape(host_shape);
}

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, const abstract::BaseShapePtr &shape,
                           const TypePtr &type, const ValuePtr &value, const ShapeVector &host_shape,
                           const UserDataPtr &user_data) {
  if (device_address != nullptr) {
    device_address_ = device_address;
    address_common_ = device_address_->address_common();
    device_address_->set_user_data(user_data);
    device_address_->set_host_shape(host_shape);
  } else {
    address_common_ = std::make_shared<AddressCommon>();
  }

  host_info_ = std::make_unique<KernelHostInfo>();
  if (type) {
    SetType(type);
  }
  if (shape) {
    // Note: for performance, the function `SetShape` uses host_info_->type_id_, so need to SetType first.
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }
}

KernelTensor::KernelTensor(const KernelTensor &other) {
  // Copy host info.
  shape_ = other.shape_ != nullptr ? other.shape_->Clone() : abstract::kNoShape;
  type_ = other.type_ != nullptr ? other.type_->Clone() : kTypeAny;
  value_ = other.value_;

  if (other.host_info_) {
    host_info_ = std::make_unique<KernelHostInfo>(*other.host_info_);
    host_info_->kernel_tensor_value_ = other.host_info_->kernel_tensor_value_ != nullptr
                                         ? std::make_shared<KernelTensorValue>(*other.host_info_->kernel_tensor_value_)
                                         : nullptr;
  }

  // Copy device info.
  task_id_on_stream_ = other.task_id_on_stream_;
  if (other.device_address_ != nullptr) {
    device_address_ = other.device_address_->CloneDeviceAddress();
    address_common_ = device_address_->address_common();
    device_address_->set_user_data(other.user_data());
    device_address_->set_heterogeneous_info(other.heterogeneous_info());
    device_address_->set_host_shape(other.host_shape());
  } else {
    address_common_ = std::make_shared<AddressCommon>(*other.address_common_);
  }
}

inline void KernelTensor::CheckHostInfoValid() {
  if (MS_UNLIKELY(!host_info_)) {
    host_info_ = std::make_unique<KernelHostInfo>();
  }
}

void KernelTensor::SetHostInfo(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  CheckHostInfoValid();
  if (type) {
    SetType(type);
  }
  if (shape) {
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }
}

void KernelTensor::SetShape(const abstract::BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  shape_ = shape;
  CheckHostInfoValid();

  // Note: for performance, the function `SetShape` uses host_info_->type_id_, so need to SetType first.
  switch (host_info_->type_id_) {
    case kObjectTypeMapTensorType:
    case kObjectTypeTensorType: {
      // The shape type check will affect the performance. The following check will be deleted after the framework is
      // stable.
      if (shape_->isa<abstract::NoShape>()) {
        address_common_->shape_vector_ = {};
      } else {
        if (!shape_->isa<abstract::TensorShape>()) {
          MS_LOG(EXCEPTION) << "Expected TensorShape for SetShape, but got: " << shape_->type_name() << ", "
                            << shape_->ToString();
        }
        address_common_->shape_vector_ = shape_->GetShapeVector();
      }

      break;
    }

    case kObjectTypeList:
    case kObjectTypeTuple: {
      if (shape->isa<abstract::DynamicSequenceShape>()) {
        address_common_->shape_vector_ = {-1};
        break;
      }
      const auto &seq_shape = shape_->cast<abstract::SequenceShapePtr>();
      if (seq_shape == nullptr) {
        MS_LOG(EXCEPTION) << "Expected SequenceShape for SetShape, but got: " << shape_->type_name() << ", "
                          << shape_->ToString();
      }
      address_common_->shape_vector_.clear();
      address_common_->shape_vector_.push_back(seq_shape->size());
      const auto &shapes = seq_shape->shape();
      if (shapes.empty()) {
        break;
      }
      const auto &element_shape = shapes[0];
      MS_EXCEPTION_IF_NULL(element_shape);
      if (element_shape->isa<abstract::TensorShape>()) {
        const ShapeVector &element_shape_vector = element_shape->GetShapeVector();
        address_common_->shape_vector_.insert(address_common_->shape_vector_.end(), element_shape_vector.begin(),
                                              element_shape_vector.end());
      } else if (element_shape->isa<abstract::SequenceShape>()) {
        const ShapeVector &element_shape_vector = GetShapeVectorByBaseShape(element_shape);
        address_common_->shape_vector_.insert(address_common_->shape_vector_.end(), element_shape_vector.begin(),
                                              element_shape_vector.end());
      }

      break;
    }

    case kTypeUnknown: {
      MS_LOG(EXCEPTION) << "Can not set shape for unknown type, please set correct type for kernel tensor first.";
    }

    default:
      MS_EXCEPTION_IF_NULL(type_);
      MS_LOG(DEBUG) << "Need not set shape for: " << type_->ToString();
  }

  // Update size_ after shape changed.
  // Note: calculate memory size should be executed after 'SetType' and 'SetShape'.
  CalculateMemSize();
}

void KernelTensor::CalculateMemSize() {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeTuple ||
      host_info_->type_id_ == kObjectTypeList) {
    // If address_common_->shape_vector_ is a dynamic shape, device_info_->size_ will be 0.
    size_t element_num = SizeOf(address_common_->shape_vector_);
    address_common_->size_ = element_num * UnitSizeInBytes(address_common_->dtype_id_);
  } else if (host_info_->type_id_ == kObjectTypeNumber) {
    address_common_->size_ = UnitSizeInBytes(address_common_->dtype_id_);
  }
}

void KernelTensor::SetShapeVector(const ShapeVector &shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    address_common_->shape_vector_ = shape_vector;
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(address_common_->shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector
                  << ", the format: " << GetFormatFromEnumToStr(address_common_->format_);
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  if (host_info_->type_id_ == kObjectTypeString) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For String Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

void KernelTensor::SetShapeVector(ShapeVector &&shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    address_common_->shape_vector_ = std::move(shape_vector);
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(address_common_->shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector
                  << ", the format: " << GetFormatFromEnumToStr(address_common_->format_);
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For String Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  if (host_info_->type_id_ == kObjectTypeString) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

using ShapeTransposeFunc = std::function<void(const ShapeVector *, ShapeVector *)>;
const ShapeVector &KernelTensor::TransposeToDeviceShape() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ != kObjectTypeTensorType) {
    MS_LOG(EXCEPTION) << "Only TensorType could transpose device shape, but got: " << TypeIdLabel(host_info_->type_id_);
  }

  static const mindspore::HashMap<mindspore::Format, ShapeTransposeFunc> shape_trans_funcs = {
    {Format::DEFAULT_FORMAT, TransposeDefaultShape},
    {Format::NCHW, TransposeNCHWShape},
    {Format::NHWC, TransposeNHWCShape}};

  auto iter = shape_trans_funcs.find(address_common_->format_);
  if (iter == shape_trans_funcs.end()) {
    MS_LOG(EXCEPTION) << "Can not find shape transpose function for format: "
                      << GetFormatFromEnumToStr(address_common_->format_);
  }

  // The shape of the device corresponding to 'address_common_->shape_vector_'. For example, if format is NHWC, the
  // shape of the device and host may be different.
  iter->second(&address_common_->shape_vector_, &host_info_->shape_vector_after_format_trasform_);
  return host_info_->shape_vector_after_format_trasform_;
}

bool KernelTensor::NeedTransposeToDeviceShape() const noexcept {
  static std::set<mindspore::Format> black_list{Format::DEFAULT_FORMAT, Format::NCHW, Format::ND, Format::NCDHW};
  auto it = black_list.find(address_common_->format_);
  return it == black_list.end();
}

const ShapeVector &KernelTensor::GetDeviceShapeVector() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (NeedTransposeToDeviceShape()) {
    std::lock_guard<std::mutex> lock(host_info_->shape_transform_mutex_);
    return TransposeToDeviceShape();
  }
  return address_common_->shape_vector_;
}

void KernelTensor::SetType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  CheckHostInfoValid();
  type_ = type;
  host_info_->type_id_ = type_->object_type();
  if (host_info_->type_id_ == kTypeUnknown) {
    host_info_->type_id_ = type_->type_id();
    MS_EXCEPTION_IF_CHECK_FAIL((host_info_->type_id_ != kTypeUnknown),
                               "Got a unknown type id, type info: " + type_->ToString());
  }

  switch (host_info_->type_id_) {
    case kObjectTypeTensorType: {
      auto tensor_type_ptr = type_->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type_ptr);
      auto element_type = tensor_type_ptr->element();
      if (element_type) {
        address_common_->dtype_id_ = element_type->type_id();
      }
    } break;

    case kObjectTypeTuple: {
      auto tuple_type = type_->cast<TuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_type);
      TypePtr element_type = nullptr;
      if (tuple_type->dynamic_len()) {
        element_type = tuple_type->dynamic_element_type();
        if (element_type == nullptr) {
          return;
        }
      } else {
        const TypePtrList &element_types = tuple_type->elements();
        if (element_types.empty()) {
          return;
        }
        element_type = element_types[0];
      }
      SetSequenceDType(element_type);
    } break;

    case kObjectTypeList: {
      auto list_type = type_->cast<ListPtr>();
      MS_EXCEPTION_IF_NULL(list_type);
      TypePtr element_type = nullptr;
      if (list_type->dynamic_len()) {
        element_type = list_type->dynamic_element_type();
        if (element_type == nullptr) {
          return;
        }
      } else {
        const TypePtrList &element_types = list_type->elements();
        if (element_types.empty()) {
          return;
        }
        element_type = element_types[0];
      }
      SetSequenceDType(element_type);
    } break;

    default:
      address_common_->dtype_id_ = type->type_id();
      MS_LOG(DEBUG) << "Set dtype for: " << type->ToString();
  }
}

void KernelTensor::SetSequenceDType(const TypePtr &element_type) {
  MS_EXCEPTION_IF_NULL(element_type);
  if (element_type->object_type() == kObjectTypeTensorType) {
    // Tensor type element.
    auto tensor_type_ptr = element_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type_ptr);
    auto tensor_element_type = tensor_type_ptr->element();
    if (tensor_element_type) {
      address_common_->dtype_id_ = tensor_element_type->type_id();
    }
  } else if (element_type->object_type() == kObjectTypeNumber) {
    // Scalar type element.
    address_common_->dtype_id_ = element_type->type_id();
  } else if (element_type->object_type() == kObjectTypeString) {
    // String type element.
    address_common_->dtype_id_ = element_type->type_id();
  } else if (element_type->object_type() == kObjectTypeTuple) {
    // Sequence type element.
    auto tuple_type = element_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    if (tuple_type->dynamic_len()) {
      if (tuple_type->dynamic_element_type() == nullptr) {
        return;
      }
      SetSequenceDType(tuple_type->dynamic_element_type());
      return;
    }
    const TypePtrList &element_types = tuple_type->elements();
    if (element_types.empty() || element_types[0] == nullptr) {
      return;
    }
    SetSequenceDType(element_types[0]);
    return;
  } else if (element_type->object_type() == kObjectTypeList) {
    // Sequence type element.
    auto list_type = element_type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(list_type);
    if (list_type->dynamic_len()) {
      if (list_type->dynamic_element_type() == nullptr) {
        return;
      }
      SetSequenceDType(list_type->dynamic_element_type());
      return;
    }
    const TypePtrList &element_types = list_type->elements();
    if (element_types.empty() || element_types[0] == nullptr) {
      return;
    }
    SetSequenceDType(element_types[0]);
    return;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported element type[" << element_type->ToString()
                      << "] to set element data type for KernelTensor.";
  }
}

std::string KernelTensor::GetStringFormat() const { return GetFormatFromEnumToStr(address_common_->format_); }

void KernelTensor::SetStringFormat(const std::string &format) {
  address_common_->format_ = GetFormatFromStrToEnum(format);
}

ValuePtr KernelTensor::GetValue() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  if (address_common_ == nullptr) {
    MS_LOG(ERROR) << "address_common_ for kernel tensor is nullptr";
  }

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (address_common_->dtype_id_ == kMetaTypeNone) {
    return kNone;
  }
  if (!SetKernelTensorValue()) {
    MS_LOG(EXCEPTION) << "Failed to get value from kernel tensor:" << this->ToString() << ", this pointer: " << this;
  }
  return host_info_->kernel_tensor_value_ != nullptr ? host_info_->kernel_tensor_value_ : value_;
}

const void *KernelTensor::GetValuePtr() {
  CheckHostInfoValid();
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (address_common_->dtype_id_ == kMetaTypeNone) {
    return nullptr;
  }
  if (!SetKernelTensorValue()) {
    MS_LOG(EXCEPTION) << "Failed to get value from kernel tensor:" << this->ToString() << ", this pointer: " << this;
  }
  MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
  return host_info_->kernel_tensor_value_->GetDataPtr();
}

bool KernelTensor::SyncDataFromDeviceToHost() const {
  // Note: must release lock when wait async resize or launch kernel finish, because the kernels' resize and launch
  // tasks which are waited maybe use this kernel's GetValue and try lock this mutex to avoid deadlock.
  host_info_->value_mutex_.unlock();
  constexpr char kWaitAsyncResizeAndLaunchFinishCallback[] = "WaitAsyncResizeAndLaunchFinish";
  static const auto wait_resize_and_launch_finish =
    KernelCallback::GetInstance().GetCallback<void>(kWaitAsyncResizeAndLaunchFinishCallback);
  if (wait_resize_and_launch_finish) {
    wait_resize_and_launch_finish();
  }
  host_info_->value_mutex_.lock();

  if (device_address_ != nullptr && device_address_->heterogeneous_info() != nullptr &&
      device_address_->heterogeneous_info()->host_ptr_ != nullptr) {
    if (!host_info_->kernel_tensor_value_) {
      host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(
        device_address_->heterogeneous_info()->host_ptr_, address_common_->size_, type_);
    } else {
      host_info_->kernel_tensor_value_->SetDataPtr(device_address_->heterogeneous_info()->host_ptr_);
      host_info_->kernel_tensor_value_->Resize(address_common_->size_);
    }
    return true;
  }

  void *device_ptr = this->device_ptr();
  if (device_ptr == nullptr) {
    MS_LOG(INFO) << "Not malloc device memory yet, sync data from device to host side failed, size: "
                 << address_common_->size_;
    return false;
  }

  MS_EXCEPTION_IF_NULL(host_info_);
  // For performance, the CPU back-end does not need to copy the device to host, and directly uses the
  // device pointer in the kernel Tensor.
  if (address_common_->device_name_ == kCPUDevice) {
    if (!host_info_->kernel_tensor_value_) {
      host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(device_ptr, address_common_->size_, type_);
    } else {
      host_info_->kernel_tensor_value_->SetDataPtr(device_ptr);
      host_info_->kernel_tensor_value_->Resize(address_common_->size_);
    }
    return true;
  }

  if (!host_info_->kernel_tensor_value_) {
    host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(address_common_->size_, type_);
  } else {
    host_info_->kernel_tensor_value_->Resize(address_common_->size_);
  }

  if (address_common_->size_ == 0) {
    return true;
  }

  void *host_ptr = host_info_->kernel_tensor_value_->GetMutableDataPtr();
  MS_EXCEPTION_IF_NULL(host_ptr);

  MS_EXCEPTION_IF_NULL(device_address_);
  if (!device_address_->SyncDeviceToHost(host_ptr, device_ptr, address_common_->size_, address_common_->device_name_,
                                         address_common_->device_id_, address_common_->format_,
                                         address_common_->shape_vector_, address_common_->stream_id_, user_data())) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed";
  }
  return true;
}

bool KernelTensor::SetKernelTensorValue() const {
  // The tensor is const value
  if (value_ != nullptr && !value_->isa<ValueAny>()) {
    if (host_info_->kernel_tensor_value_ == nullptr) {
      host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
    }
    return true;
  }

  // The tensor is variable value that is set in user_data.
  if (user_data() != nullptr) {
    auto var_host_value = user_data()->get<std::pair<ValuePtr, bool>>("variable_host_value");
    if (var_host_value != nullptr) {
      if (var_host_value->second) {
        MS_LOG(DEBUG) << "Set kernel_tensor_value from host value in user data: " << var_host_value->first->ToString();
        host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(var_host_value->first);
        var_host_value->second = false;
      }
      return true;
    }
    // Set user data for PyExecute infer.
    if (user_data()->has(kGetValueByUserDataHandler)) {
      const auto &handler = user_data()->get<ValuePtr (*)(const UserDataPtr &)>(kGetValueByUserDataHandler);
      if (handler != nullptr) {
        auto value = (*handler)(user_data());
        if (value != nullptr) {
          host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value);
          return true;
        }
      }
    }
  }

  // Sync value data from device.
  if (!SyncDataFromDeviceToHost()) {
    MS_LOG(INFO) << "Sync data from device to host side failed";
    return false;
  }
  return true;
}

bool KernelTensor::IsDynamicShape() const {
  const auto &shape = this->GetShapeVector();
  return std::any_of(shape.cbegin(), shape.cend(), [](auto i) { return i < 0; });
}

ShapeVector KernelTensor::GetMaxShape() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ != kObjectTypeTensorType) {
    return {};
  }
  if (shape_ == nullptr || !shape_->isa<abstract::Shape>()) {
    return {};
  }

  return shape_->cast<abstract::ShapePtr>()->max_shape();
}

const DeviceAddressPtr &KernelTensor::device_address() const { return device_address_; }
void KernelTensor::set_device_address(const DeviceAddressPtr &device_address) {
  device_address_ = device_address;
  if (device_address_ != nullptr) {
    address_common_ = device_address_->address_common();
  }
}
}  // namespace mindspore::kernel
