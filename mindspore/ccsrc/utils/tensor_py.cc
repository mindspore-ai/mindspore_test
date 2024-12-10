/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
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

#include "include/common/utils/tensor_py.h"

#include "utils/log_adapter.h"

namespace mindspore {
namespace tensor {
TensorPy::TensorPy(const BaseTensorPtr &input) { tensor_ = input; }

TensorPy::TensorPy(const TensorPtr &input) { tensor_ = input; }

TensorPy::TensorPy(int64_t input, const TypePtr &data_type) { tensor_ = std::make_shared<Tensor>(input, data_type); }

TensorPy::TensorPy(int32_t input, const TypePtr &data_type) { tensor_ = std::make_shared<Tensor>(input, data_type); }

TensorPy::TensorPy(int16_t input, const TypePtr &data_type) { tensor_ = std::make_shared<Tensor>(input, data_type); }

TensorPy::TensorPy(int8_t input, const TypePtr &data_type) { tensor_ = std::make_shared<Tensor>(input, data_type); }

TensorPy::TensorPy(const std::vector<int64_t> &input, const TypePtr &data_type) {
  tensor_ = std::make_shared<Tensor>(input, data_type);
}

TensorPy::TensorPy(const std::vector<int32_t> &input, const TypePtr &data_type) {
  tensor_ = std::make_shared<Tensor>(input, data_type);
}

TensorPy::TensorPy(const std::vector<double> &input, const TypePtr &data_type) {
  tensor_ = std::make_shared<Tensor>(input, data_type);
}

TensorPy::TensorPy(const std::vector<float> &input, const TypePtr &data_type) {
  tensor_ = std::make_shared<Tensor>(input, data_type);
}

TensorPy::TensorPy(const TensorPy &input)
    : init_finished_flag_(input.init_finished_flag_),
      const_arg_flag_(input.const_arg_flag_),
      virtual_flag_(input.virtual_flag_),
      ms_parameter_output_(input.ms_parameter_output_),
      initializer_(input.initializer_),
      parent_tensor_(input.parent_tensor_),
      index_of_parent_(input.index_of_parent_),
      symbolic_shape_(input.symbolic_shape_),
      device_(input.device_),
      tensor_(input.tensor_),
      flatten_tensor_(input.flatten_tensor_) {}

TensorPy::TensorPy(TypeId data_type, const ShapeVector &shape) { tensor_ = std::make_shared<Tensor>(data_type, shape); }

bool TensorPy::IsInitFinished() { return init_finished_flag_; }

void TensorPy::SetInitFinished(bool flag) { init_finished_flag_ = flag; }

bool TensorPy::IsConstArg() { return const_arg_flag_; }

void TensorPy::SetConstArg(bool flag) { const_arg_flag_ = flag; }

bool TensorPy::IsVirtual() { return virtual_flag_; }

void TensorPy::SetVirtualFlag(bool flag) { virtual_flag_ = flag; }

const py::object TensorPy::GetInitializer() const {
  if (!initializer_.check() || initializer_.is_none()) {
    return py::none();
  }
  return initializer_;
}

void TensorPy::SetInitializer(const py::object &init) { initializer_ = init; }

const std::string TensorPy::GetDevice() const { return device_; }

void TensorPy::SetDevice(const std::string &dev) { device_ = dev; }

const TensorPtr TensorPy::GetTensor() const {
  MS_EXCEPTION_IF_NULL(tensor_);
  TensorPtr tensor = std::dynamic_pointer_cast<Tensor>(tensor_);
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}

const BaseTensorPtr TensorPy::GetBaseTensor() const {
  MS_EXCEPTION_IF_NULL(tensor_);
  return tensor_;
}

const py::object TensorPy::GetParentTensor() {
  if (!parent_tensor_.check() || parent_tensor_.is_none()) {
    return py::none();
  }
  return parent_tensor_;
}

void TensorPy::SetParentTensor(const py::object &parent) { parent_tensor_ = parent; }

const py::object TensorPy::GetIndexOfParent() {
  if (!index_of_parent_.check() || index_of_parent_.is_none()) {
    return py::none();
  }
  return index_of_parent_;
}

void TensorPy::SetIndexOfParent(const py::object &index) { index_of_parent_ = index; }

const py::object TensorPy::GetSymbolicShape() const {
  if (!symbolic_shape_.check() || symbolic_shape_.is_none()) {
    return py::none();
  }
  return symbolic_shape_;
}

void TensorPy::SetSymbolicShape(const py::object &symbolic) { symbolic_shape_ = symbolic; }

py::tuple TensorPy::GetPyTupleShape() {
  auto &shape = GetBaseTensor()->shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::int_ TensorPy::GetPyItemSize() { return GetBaseTensor()->data().itemsize(); }

py::int_ TensorPy::GetPyNBytes() { return GetBaseTensor()->data().nbytes(); }

static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

py::tuple TensorPy::GetPyTupleStrides() {
  auto tensor = GetBaseTensor();
  std::vector<ssize_t> shape(tensor->shape().begin(), tensor->shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor->data().itemsize());
  py::tuple py_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    py_strides[i] = py::int_(strides[i]);
  }
  return py_strides;
}

TypePtr TensorPy::GetDtype() const { return GetBaseTensor()->Dtype(); }

TypePtr TensorPy::SetDtype(const TypePtr type) { return GetBaseTensor()->SetDtype(type); }

TypeId TensorPy::GetDataType() const { return GetBaseTensor()->data_type(); }

const ShapeVector &TensorPy::GetShape() const { return GetBaseTensor()->shape(); }

bool TensorPy::IsInit() const { return GetTensor()->is_init(); }

void TensorPy::SetInitFlag(bool flag) { GetTensor()->set_init_flag(flag); }

bool TensorPy::IsAdapter() const { return GetTensor()->is_adapter(); }

void TensorPy::SetAdapterFlag(bool flag) { GetTensor()->set_adapter_flag(flag); }

void TensorPy::SetShape(const ShapeVector &shape) { GetBaseTensor()->set_shape(shape); }

bool TensorPy::IsPersistentData() const { return GetTensor()->is_persistent_data(); }

int TensorPy::DataDim() const { return GetBaseTensor()->DataDim(); }

TensorPy &TensorPy::AssignValue(const TensorPy &tensorpy) {
  auto tensor = tensorpy.GetTensor().get();
  GetTensor()->AssignValue(*tensor);
  return *this;
}

bool TensorPy::Offload(const std::string &file_path) { return GetTensor()->Offload(file_path); }

const std::string TensorPy::GetOffloadFilePath() const { return GetTensor()->GetOffloadFilePath(); }

void TensorPy::SetCastDtype(const TypePtr &dtype) { GetTensor()->set_cast_dtype(dtype); }

void TensorPy::DataSync(bool need_wait) const { GetBaseTensor()->data_sync(need_wait); }

void TensorPy::ExecuteLazyTask() const { GetBaseTensor()->ExecuteLazyTask(); }

bool TensorPy::IsContiguous() const { return GetBaseTensor()->is_contiguous(); }

std::vector<int64_t> TensorPy::GetStride() const { return GetBaseTensor()->stride(); }

const int64_t TensorPy::GetStorageOffset() const { return GetBaseTensor()->storage_offset(); }

std::string TensorPy::ToString() const { return GetBaseTensor()->ToString(); }

std::string TensorPy::ToStringRepr() const { return GetBaseTensor()->ToStringRepr(); }

bool TensorPy::CheckStub() { return Tensor::CheckStub(); }

ParamInfoPtr TensorPy::GetParamInfo() const { return GetBaseTensor()->param_info(); }

void TensorPy::SetParamInfo(const ParamInfoPtr &param_info) {
  MS_EXCEPTION_IF_NULL(tensor_);
  GetBaseTensor()->set_param_info(param_info);
}

const TensorPyPtr TensorPy::GetFlattenTensor() { return flatten_tensor_; }

void TensorPy::SetFlattenTensor(const TensorPyPtr tensor) {
  if (tensor == nullptr) {
    return;
  }
  flatten_tensor_ = tensor;
}

bool TensorPy::IsFlattened(const TensorPyPtrList &tensorpys) {
  TensorPtrList tensors;
  (void)std::transform(tensorpys.begin(), tensorpys.end(), std::back_inserter(tensors),
                       [](const TensorPyPtr &p) { return p->GetTensor(); });
  return Tensor::IsFlattened(tensors);
}

TensorPyPtrList TensorPy::FlattenTensors(const TensorPyPtrList &tensorpys, size_t fusion_size) {
  TensorPtrList tensors;
  (void)std::transform(tensorpys.begin(), tensorpys.end(), std::back_inserter(tensors),
                       [](const TensorPyPtr &p) { return p->GetTensor(); });
  TensorPtrList out_tensors = Tensor::FlattenTensors(tensors, fusion_size);
  TensorPyPtrList out;
  (void)std::transform(out_tensors.begin(), out_tensors.end(), std::back_inserter(out),
                       [&tensorpys](const TensorPtr &p) {
                         auto tensorpy = std::make_shared<TensorPy>(p);
                         for (auto t : tensorpys) {
                           auto flatten = Tensor::GetFlattenedTensor(t->GetTensor());
                           if (p == flatten) {
                             t->SetFlattenTensor(tensorpy);
                           }
                         }
                         return tensorpy;
                       });
  return out;
}

TensorPyPtrList TensorPy::GetFlattenedTensors(const TensorPyPtrList &tensorpys) {
  TensorPtrList tensors;
  (void)std::transform(tensorpys.begin(), tensorpys.end(), std::back_inserter(tensors),
                       [](const TensorPyPtr &p) { return p->GetTensor(); });
  TensorPtrList out_tensors = Tensor::GetFlattenedTensors(tensors);
  if (out_tensors.empty()) {
    return {};
  }

  // Use std::map to keep order by type id.
  std::map<TypeId, OrderedSet<TensorPyPtr>> chunk_map;
  for (auto &tensorpy : tensorpys) {
    auto owner_tensorpy = tensorpy->GetFlattenTensor();
    auto get_normalize_type = [](TypeId id) {
      if (id == kNumberTypeFloat) {
        // kNumberTypeFloat is an alias of kNumberTypeFloat32.
        return kNumberTypeFloat32;
      }
      return id;
    };
    auto chunk_dtype = get_normalize_type(tensorpy->GetDataType());
    chunk_map[chunk_dtype].add(owner_tensorpy);
  }
  // Generate result tensorpy list.
  TensorPyPtrList result_tensorpys;
  for (auto &entry : chunk_map) {
    auto &chunk_tensors = entry.second;
    (void)result_tensorpys.insert(result_tensorpys.end(), chunk_tensors.begin(), chunk_tensors.end());
  }
  return result_tensorpys;
}

size_t TensorPy::GetFusionSize(const TensorPyPtrList &flat_tensorpys) {
  TensorPtrList tensors;
  (void)std::transform(flat_tensorpys.begin(), flat_tensorpys.end(), std::back_inserter(tensors),
                       [](const TensorPyPtr &p) { return p->GetTensor(); });
  return Tensor::GetFusionSize(tensors);
}

const size_t TensorPy::GetDataSize() const { return GetBaseTensor()->DataSize(); }

void *TensorPy::GetTensorDataObject() const { return GetBaseTensor()->data_c(); }

const DeviceSyncPtr TensorPy::GetDeviceAddress() const { return GetBaseTensor()->device_address(); }

bool TensorPy::IsMSParameterOutput() const { return ms_parameter_output_; }

void TensorPy::SetMSParameterOutput(bool flag) { ms_parameter_output_ = flag; }

bool TensorPy::HasAutoGrad() const { return GetBaseTensor()->HasAutoGrad(); }

bool TensorPy::NeedContiguous() const { return GetBaseTensor()->NeedContiguous(); }

bool IsTensorPy(const py::handle &obj) { return py::isinstance<TensorPy>(obj); }

const TensorPyPtr ConvertToTensorPy(const py::handle &obj) { return obj.cast<TensorPyPtr>(); }

const TensorPtr ConvertToTensor(const py::handle &obj) {
  if (IsTensorPy(obj)) {
    TensorPyPtr tensorpy = ConvertToTensorPy(obj);
    return tensorpy->GetTensor();
  }

  return nullptr;
}

}  // namespace tensor
}  // namespace mindspore
