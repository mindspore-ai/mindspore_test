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

namespace mindspore {
namespace tensor {
// TensorSubData is the base class to provide tensor data as a segment from an owner tensor data.
class TensorSubData : public TensorData {
 public:
  TensorSubData(const TensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : data_owner_(data_owner), data_offset_(offset), data_size_(data_size), ndim_(ndim) {}
  TensorSubData(const BaseTensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
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
  const BaseTensorPtr &GetOwner() const { return data_owner_; }

  // Data offset in bytes.
  size_t data_offset() const { return data_offset_; }

 protected:
  const BaseTensorPtr data_owner_;
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
  TensorSubDataImpl(const BaseTensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : TensorSubData(data_owner, offset, data_size, ndim) {}

  ~TensorSubDataImpl() override = default;

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override {
    TensorStringifier<T> stringifier{static_cast<const T *>(const_data()), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }
};

TensorDataPtr MakeTensorSubData(const BaseTensorPtr &owner, size_t offset, const TensorDataPtr &data) {
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
  size_t size{0};                      // chunk size in the number of elements.
  size_t bytes{0};                     // chunk size in bytes.
  std::vector<BaseTensorPtr> tensors;  // tensors belong to this chunk.
};

static TypeId normalize_type(TypeId type_id) {
  if (type_id == kNumberTypeFloat) {
    // kNumberTypeFloat is an alias of kNumberTypeFloat32.
    return kNumberTypeFloat32;
  }
  return type_id;
}

Tensor::Tensor(const Tensor &tensor)
    : BaseTensor(tensor),
      init_flag_(tensor.init_flag_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      pin_mem_register_(tensor.pin_mem_register_),
      compression_type_(tensor.compression_type_),
      tensor_name_(tensor.tensor_name_),
      device_info_(tensor.device_info_),
      copy_done_flag_(tensor.copy_done_flag_) {}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : BaseTensor(tensor, data_type),
      init_flag_(tensor.init_flag_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      pin_mem_register_(tensor.pin_mem_register_),
      compression_type_(tensor.compression_type_),
      tensor_name_(tensor.tensor_name_),
      device_info_(tensor.device_info_),
      copy_done_flag_(tensor.copy_done_flag_) {}

Tensor::Tensor(const BaseTensor &tensor, TypeId data_type) : BaseTensor(tensor, data_type) {}

Tensor::Tensor(const BaseTensor &base_tensor) : BaseTensor(base_tensor) {}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (this == &tensor) {
    return *this;
  }
  BaseTensor::operator=(tensor);
  init_flag_ = tensor.init_flag_;
  need_release_device_mem_ = tensor.need_release_device_mem_;
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
  device_info_ = tensor.device_info_;
  copy_done_flag_ = tensor.copy_done_flag_;
  return *this;
}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data) : BaseTensor(data_type, shape, data) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape) : BaseTensor(data_type, shape) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : BaseTensor(data_type, shape, data, data_len) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type)
    : BaseTensor(data_type, shape, data, src_data_type) {}

Tensor::Tensor(const std::vector<int64_t> &input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(const std::vector<int32_t> &input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(const std::vector<float> &input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(int64_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(int32_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(int16_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(int8_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(double input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(float input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(float16 input, const TypePtr &data_type) : BaseTensor(input, data_type) {}
#ifndef KERNEL_EXECUTOR_ANDROID
Tensor::Tensor(bfloat16 input, const TypePtr &data_type) : BaseTensor(input, data_type) {}
#endif
Tensor::Tensor(uint64_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(uint32_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(uint16_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(uint8_t input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(bool input, const TypePtr &data_type) : BaseTensor(input, data_type) {}

Tensor::Tensor(TypeId data_type, size_t data_size) : BaseTensor(data_type, data_size) {}

Tensor::Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
               TensorCompressionType compression_type)
    : BaseTensor(origin_data_type, shape, compression_data_size, compression_type) {
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
  return (&tensor == this || (BaseTensor::operator==(tensor) && data_ == tensor.data_));
}

// Assign value to this tensor.
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    BaseTensor::AssignValue(tensor);
    device_info_ = tensor.device_info_;
    need_release_device_mem_ = tensor.need_release_device_mem_;

    // Need execute callback when update host value of Tensor.
    ExecuteUpdateValueCallback();
  }
  return *this;
}

abstract::AbstractBasePtr Tensor::ToAbstract() { return BaseTensor::ToAbstract()->cast<abstract::AbstractTensorPtr>(); }

void Tensor::data_sync(bool need_wait) const { BaseTensor::data_sync(need_wait); }

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
  MS_EXCEPTION_IF_NULL(data_);
  if (data_->is_sub_data()) {
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), IntToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  if (size != 0 && !device_sync->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
    MS_LOG(INTERNAL_EXCEPTION) << "SyncDeviceToHost failed.";
  }
  sync_status_ = kNeedSyncHostToDevice;
}

bool Tensor::Offload(const std::string &file_path) {
  if (file_path.empty()) {
    return false;
  }

  auto fs = mindspore::system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  MS_EXCEPTION_IF_NULL(data_);
  auto data_ptr = data_->data();
  auto file = fs->CreateWriteFile(file_path);
  MS_EXCEPTION_IF_NULL(file);
  TempFileManager::GetInstance().Register(file_path);
  bool success = file->PWrite(data_ptr, LongToSize(data_->nbytes()), 0);
  if (!file->Close()) {
    MS_LOG(WARNING) << "Close tensor file: " << file_path << " failed!";
  }
  if (!success) {
    MS_LOG(WARNING) << "Tensor write data to file: " << file_path << " failed!";
    return false;
  }

  if (file_path == GetOffloadFilePath()) {
    data_->set_file_path("");
  }

  data_ = tensor::MakeTensorData(data_type_, shape_);
  MS_EXCEPTION_IF_NULL(data_);
  data_->set_file_path(file_path);
  return true;
}

const std::string Tensor::GetOffloadFilePath() const {
  if (data_ == nullptr) {
    return "";
  }
  return data_->file_path();
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
    auto tensor_bytes = static_cast<size_t>(tensor->data().nbytes());
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
    auto tensor_bytes = static_cast<size_t>(tensor->data().nbytes());
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
    return indptr_->data().nbytes();
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_->data().nbytes();
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_->data().nbytes();
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
}  // namespace tensor
}  // namespace mindspore
