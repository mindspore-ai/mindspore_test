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

#ifndef MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_
#define MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_

#include <cstddef>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <algorithm>
#include <functional>
#include "abstract/abstract_value.h"
#include "mindapi/base/format.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/api/format.h"
#include "include/backend/visible.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/kernel_tensor_value.h"
#include "common/kernel_visible.h"
#include "common/device_address.h"

namespace mindspore {
namespace kernel {
using abstract::AbstractBase;
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;

template <typename T>
struct ValidContainerChecker : std::false_type {};

// A ValidContainerChecker's specialization to detect whether the type is std::vector whose element is scalar.
template <typename... Args>
struct ValidContainerChecker<std::vector<Args...>> : std::true_type {};

// A ValidContainerChecker's specialization to detect whether the type is std::string.
template <>
struct ValidContainerChecker<std::string> : std::true_type {};

// A wrapper used to check the types std::string and std::vector.
template <typename T>
struct IsValidContainer {
  static constexpr bool value = ValidContainerChecker<std::decay_t<T>>::value;
};

// Used to encapsulate host-side related data structures in KernelTensor.
struct KernelHostInfo {
  KernelHostInfo() = default;

  KernelHostInfo(const KernelHostInfo &other);
  KernelHostInfo &operator=(const KernelHostInfo &other) = delete;

  // The shape vector transformed according `shape_vector_` and `format_` is generally used on the operator side.
  // Operators on different platforms may require different format and shape information.
  ShapeVector shape_vector_after_format_trasform_{};

  // Make shape transform related interfaces thread-safe.
  std::mutex shape_transform_mutex_;

  // The object enum type id of the KernelTensor.
  TypeId type_id_{kTypeUnknown};

  // Saves the contents after the value is converted to continuous memory storage.
  KernelTensorValuePtr kernel_tensor_value_{nullptr};

  // Make GetValue related interfaces thread-safe.
  std::mutex value_mutex_;
};

struct Address {
  Address() : addr(nullptr), size(0) {}
  Address(void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;

// KernelTensor is used to express input and output parameters of kernels.
// KernelTensor is a generalized Tensor semantics, which can represent not only Tensor, but also the meta-information
// of Scalar, Tuple, List and other data structures. It saves the shape, type, value and format information required by
// operators Infer and Launch, and provides related Get/Set interfaces.
class OPS_KERNEL_COMMON_API KernelTensor : public AbstractBase {
 public:
  using Deleter = PointerRefCount::Deleter;

  KernelTensor();
  ~KernelTensor() = default;

  // Constructor of KernelTensor by shape, type, value.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Constructor of KernelTensor by device info.
  KernelTensor(const DeviceAddressPtr &device_address, TypeId dtype_id, const ShapeVector &host_shape);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const DeviceAddressPtr &device_address, const abstract::BaseShapePtr &shape, const TypePtr &type,
               const ValuePtr &value, void *device_ptr, size_t size, const std::string &format, TypeId dtype_id,
               const ShapeVector &host_shape, const string &device_name, uint32_t device_id);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const DeviceAddressPtr &device_address, const abstract::BaseShapePtr &shape, const TypePtr &type,
               const ValuePtr &value, const ShapeVector &host_shape, const UserDataPtr &user_data = nullptr);

  explicit KernelTensor(const DeviceAddressPtr &device_address)
      : KernelTensor(device_address, nullptr, nullptr, nullptr, {}) {}

  KernelTensor(const KernelTensor &other);
  KernelTensor &operator=(const KernelTensor &) = delete;

  MS_DECLARE_PARENT(KernelTensor, AbstractBase);

  std::string ToString() const {
    std::stringstream ofs;
    ofs << this << " shape:" << (GetShape() == nullptr ? "null" : GetShape()->ToString())
        << " type:" << (GetType() == nullptr ? "null" : GetType()->ToString())
        << " value:" << (value_ == nullptr ? "null" : value_->ToString());
    if (device_address_ != nullptr) {
      return ofs.str() + " device address:" + device_address_->ToString();
    }
    if (address_common_ != nullptr) {
      return ofs.str() + " address common:" + address_common_->ToString();
    }
    return ofs.str() + "device address:0";
  }

  // Get the base shape for Tensor/Sequence/Scalar.
  abstract::BaseShapePtr GetShape() const override { return shape_; }

  // Set the base shape for Tensor/Sequence/Scalar.
  // Note: for performance, the function `SetShape` uses type_id_, so need to SetType first.
  void SetShape(const abstract::BaseShapePtr &shape);

  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const { return address_common_->shape_vector_; }

  // Set the shape vector for Tensor/Sequence/Scalar.
  void SetShapeVector(const ShapeVector &shape_vector);

  // Set the shape vector for Tensor/Sequence/Scalar with rvalue.
  void SetShapeVector(ShapeVector &&shape_vector);

  // Get the device shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetDeviceShapeVector() const;

  // Get host shape for KernelTensor.
  const ShapeVector &host_shape() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->host_shape();
  }

  // Set host shape for KernelTensor.
  void set_host_shape(const ShapeVector &host_shape) {
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->set_host_shape(host_shape);
  }

  // Get the object type of the KernelTensor.
  TypePtr GetType() const override { return type_; }

  // Set the type for the KernelTensor.
  void SetType(const TypePtr &type);

  // Check whether the host info exists.
  bool host_info_exist() const { return host_info_ != nullptr; }

  // Set host info after construct
  void SetHostInfo(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Get the object enum type id of the KernelTensor.
  TypeId type_id() const {
    MS_EXCEPTION_IF_NULL(host_info_);
    return host_info_->type_id_;
  }

  // Get the data enum type id of the KernelTensor.
  TypeId dtype_id() const { return address_common_->dtype_id_; }

  // Set the data enum type id of the KernelTensor.
  void set_dtype_id(TypeId dtype_id) { address_common_->dtype_id_ = dtype_id; }

  // Set the value for the KernelTensor.
  void SetValue(const ValuePtr &value) { value_ = value; }

  // Get the value of the KernelTensor.
  ValuePtr GetValue() const override;

  // Get the address of the value converted to continuous memory storage.
  const void *GetValuePtr();

  // Get the value in KernelTensor, return it if there is specific value, otherwise throw an exception.
  template <typename T>
  T GetValueWithCheck() {
    auto value_opt = GetValue<T>();
    if (!value_opt.has_value()) {
      MS_LOG(EXCEPTION)
        << "Get value failed, there is no any value in KernelTensor."
           "Here are the possible reasons:"
           "1. When the operator KernelMod is registered, the data type is not correct, such as Scalar or Tuple, "
           "but is registered as Tensor."
           "2. If the KernelMod is registered correctly, it may be an attempt to GetValue the output of the "
           "previous operator. During compilation, the output of the operator has no value. You can check the ir "
           "file to see if the input for the current operator value is from an operator.";
    }
    return value_opt.value();
  }

  // Get the scalar value store in KernelTensor if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    MS_EXCEPTION_IF_NULL(host_info_);
    std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (address_common_->dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid scalar value.";
      return std::nullopt;
    }

    if (!SetKernelTensorValue()) {
      return std::nullopt;
    }
    MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
    MS_EXCEPTION_IF_CHECK_FAIL((host_info_->kernel_tensor_value_->GetDataSize() == sizeof(T)),
                               "The data size in kernel tensor value which contains a scalar [" +
                                 std::to_string(host_info_->kernel_tensor_value_->GetDataSize()) +
                                 "] is not equal to the data type size [" + std::to_string(sizeof(T)) + "]");

    const T *data_ptr = reinterpret_cast<const T *>(host_info_->kernel_tensor_value_->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    return *data_ptr;
  }

  // Get the std::vector/std::string value store in KernelTensor if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<IsValidContainer<T>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    if (!std::is_scalar_v<typename T::value_type>) {
      MS_LOG(EXCEPTION) << "The element of std::vector to get kernel tensor's value should be scalar type.";
    }
    MS_EXCEPTION_IF_NULL(host_info_);
    std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (address_common_->dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid value for vector or string.";
      return std::nullopt;
    }

    if (!SetKernelTensorValue()) {
      return std::nullopt;
    }
    MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
    size_t element_num = host_info_->kernel_tensor_value_->GetDataSize() / sizeof(typename T::value_type);
    if (element_num == 0) {
      return T();
    }
    const typename T::value_type *data_ptr =
      reinterpret_cast<const typename T::value_type *>(host_info_->kernel_tensor_value_->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);

    return T(data_ptr, data_ptr + element_num);
  }

  // Get the value stored in KernelTensor for type which is not scalar, std::vector or std::string if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<!IsValidContainer<T>::value && !std::is_pointer_v<T> &&
                                                !std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    if (address_common_->dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid value.";
      return std::nullopt;
    }
    if (value_ && !value_->isa<ValueAny>()) {
      return mindspore::GetValue<T>(value_);
    }
    return std::nullopt;
  }

  // Get the value in KernelTensor, return it if there is specific value, otherwise throw an exception.
  template <typename T>
  std::optional<T> GetOptionalValueWithCheck() {
    if (value_ && value_->isa<None>()) {
      return std::nullopt;
    }
    return GetValueWithCheck<T>();
  }

  // Get the data format.
  mindspore::Format format() const { return address_common_->format_; }

  // Set the data format.
  void set_format(mindspore::Format format) { address_common_->format_ = format; }

  // Get the data format of string type.
  std::string GetStringFormat() const;

  // Set the data format of string type.
  void SetStringFormat(const std::string &format);

  // Get pointer and reference count.
  const PointerRefCountPtr &pointer_ref_count() const { return address_common_->pointer_ref_count_; }

  // Set pointer and reference count.
  void set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt) {
    address_common_->pointer_ref_count_ = ptr_ref_cnt;
  }

  void set_ref_count_without_hold(const PointerRefCountPtr &ptr_ref_cnt) {
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

  //  Set the pointer and reference count to nullptr, resource reclaiming of the device pointer is automatically
  //  released.
  void ReleaseDeviceRes() { address_common_->pointer_ref_count_ = nullptr; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { address_common_->pointer_ref_count_->set_deleter(deleter); }

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const { return address_common_->pointer_ref_count_->ptr(); }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr) { address_common_->pointer_ref_count_->set_ptr(ptr); }

  // Get the memory size in byte of the KernelTensor.
  size_t size() const { return address_common_->size_; }

  // Set the memory size in byte of the KernelTensor.
  void set_size(size_t size) { address_common_->size_ = size; }

  // Get device target name, such "GPU","Ascend".
  const std::string &device_name() const { return address_common_->device_name_; }

  // Set device target name, such "GPU","Ascend".
  void set_device_name(const std::string &device_name) { address_common_->device_name_ = device_name; }

  // Get device id.
  uint32_t device_id() const { return address_common_->device_id_; }

  // Set device id.
  void set_device_id(uint32_t device_id) { address_common_->device_id_ = device_id; }

  // Get logical stream id.
  uint32_t stream_id() const { return address_common_->stream_id_; }

  // Set logical stream id.
  void set_stream_id(uint32_t stream_id) { address_common_->stream_id_ = stream_id; }

  // Get task id on stream.
  std::shared_ptr<int64_t> task_id_on_stream() const { return task_id_on_stream_; }

  // Set task id on stream.
  void set_task_id_on_stream(const std::shared_ptr<int64_t> &task_id_on_stream) {
    task_id_on_stream_ = task_id_on_stream;
  }

  bool managed_by_somas() const { return address_common_->managed_by_somas_; }

  void set_managed_by_somas(bool managed_by_somas) { address_common_->managed_by_somas_ = managed_by_somas; }

  // Get user data maintained by the KernelTensor.
  UserDataPtr user_data() const {
    if (device_address_ == nullptr) {
      return nullptr;
    }
    return device_address_->user_data();
  }

  // Set user data to the KernelTensor.
  void set_user_data(const UserDataPtr &user_data) {
    if (device_address_ == nullptr) {
      return;
    }
    device_address_->set_user_data(user_data);
  }

  HeterogeneousInfoPtr heterogeneous_info() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->heterogeneous_info();
  }

  void set_heterogeneous_info(HeterogeneousInfoPtr hete_info) {
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->set_heterogeneous_info(hete_info);
  }

  // Clone a new KernelTensor from this.
  std::shared_ptr<KernelTensor> CloneKernelTensor() { return std::make_shared<KernelTensor>(*this); }

  // Check whether the shape is dynamic shape(contains dim which is less than 0).
  bool IsDynamicShape() const;

  // Check whether the KernelTensor is from a constant variable(such as ValueNode).
  inline bool IsConstValue() const { return (value_ != nullptr) && !(value_->isa<ValueAny>()); }

  // The following four methods are only used in the Lite framework.
  // Get the device data address(pointer and size).
  AddressPtr GetData() const { return data_; }
  // Set the device data address(pointer and size).
  void SetData(const AddressPtr &data) { data_ = data; }
  // Get the host data address(pointer and size).
  AddressPtr GetHostData() const { return host_data_; }
  // Set the host data address(pointer and size).
  void SetHostData(const AddressPtr &data) { host_data_ = data; }

  // max shape is only used in compute-depended ops
  ShapeVector GetMaxShape() const;

  const TensorStorageInfoPtr tensor_storage_info() const { return address_common_->tensor_storage_info_; }
  void set_tensor_storage_info(const TensorStorageInfoPtr &storage_info) {
    if (storage_info) {
      auto ori_shape = storage_info->ori_shape;
      auto type_size = GetTypeByte(TypeIdToType(dtype_id()));
      storage_info->ori_size =
        std::accumulate(ori_shape.begin(), ori_shape.end(), type_size, std::multiplies<size_t>());
    }
    address_common_->tensor_storage_info_ = storage_info;
  }

  const device::DeviceType GetDeviceType() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->GetDeviceType();
  }

  size_t GetSize() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->GetSize();
  }

  // The interface of flag.
  size_t flag() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->flag();
  }
  size_t original_ref_count() const { return address_common_->pointer_ref_count_->original_ref_count(); }
  size_t ref_count() const { return address_common_->pointer_ref_count_->ref_count(); }
  int32_t dynamic_ref_count() const { return address_common_->pointer_ref_count_->dynamic_ref_count(); }
  size_t new_ref_count() const { return address_common_->pointer_ref_count_->new_ref_count(); }

  const DeviceAddressPtr &device_address() const;
  void set_device_address(const DeviceAddressPtr &device_address);
  const AddressCommonPtr address_common() const { return address_common_; }

  // For output of pyexecute kernel, the input data is stored in user data and the handler is used to sync data from
  // user data to device ptr.
  bool need_sync_user_data() {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->need_sync_user_data();
  }
  void set_need_sync_user_data(bool need_sync_user_data) {
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->set_need_sync_user_data(need_sync_user_data);
  }

  void Swap(KernelTensor *other) {
    MS_EXCEPTION_IF_NULL(other);
    auto other_device_address = other->device_address().get();
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->Swap(other_device_address);
    set_task_id_on_stream(other->task_id_on_stream());
  }

  // Return whether KernelTensor has a valid ptr.
  bool IsPtrValid() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->IsPtrValid();
  }

  void ClearUserData() {
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->ClearUserData();
  }

 private:
  // This is a deprecated function in base class.
  BaseShapePtr BuildShape() const override {
    MS_LOG(EXCEPTION) << "Call deprecated function: BuildShape, Please use GetShape instead of BuildShape in "
                         "operators' infer functions in the `core/ops` directory.";
  }

  // This is a deprecated function in base class
  TypePtr BuildType() const override {
    MS_LOG(EXCEPTION) << "Call deprecated function: BuildType, Please use GetType instead of BuildType in "
                         "operators' infer functions in the `core/ops` directory.";
  }

  // Set the element data type to KernelTensor for Sequence type(Tuple or List).
  void SetSequenceDType(const TypePtr &element_type);

  // Synchronize value data from device to host side.
  bool SyncDataFromDeviceToHost() const;

  // Update the kernel_tensor_value from host or device data.
  bool SetKernelTensorValue() const;

  // Calculate memory size need by the KernelTensor.
  void CalculateMemSize();

  // Check whether need to transpose host infer shape to device shape.
  bool NeedTransposeToDeviceShape() const noexcept;

  // Transpose host infer shape to device shape according format.
  const ShapeVector &TransposeToDeviceShape() const;

  // If host info is not initialized in the constructor, initialize it when you need it, making sure that host info is
  // not empty when used.
  void CheckHostInfoValid();

  // The host-side related data in KernelTensor.
  // Note: To improve the performance of constructing KernelTensor, allow some constructors not to initialize host
  // info. If host info is not initialized in the constructor, it can be initialized when it is needed.
  std::unique_ptr<KernelHostInfo> host_info_{nullptr};

  // The launch index on stream managed by framework.
  std::shared_ptr<int64_t> task_id_on_stream_{nullptr};

  // The following two variables are only used in the Lite framework.
  // Device data address.
  AddressPtr data_{nullptr};
  // Host data address.
  AddressPtr host_data_{nullptr};

  // device address info
  DeviceAddressPtr device_address_{nullptr};
  AddressCommonPtr address_common_{nullptr};
};
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_
