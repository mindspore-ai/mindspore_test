/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "ir/value.h"
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils_py.h"
#include "debug/profiler/profiler.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/pyobj_manager.h"

namespace mindspore {
namespace tensor {
PyTypeObject *TensorPy_Type;

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
      flatten_tensor_(input.flatten_tensor_) {
  tensor_ = input.GetBaseTensor();
}

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
  if (tensor_ == nullptr) {
    const_cast<BaseTensorPtr &>(tensor_) = GetBaseTensor();
  }
  TensorPtr tensor = std::dynamic_pointer_cast<Tensor>(tensor_);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Copy tensor " << tensor_->id() << " and detach!";
    auto new_tensor = std::make_shared<Tensor>(*tensor_);
    const_cast<BaseTensorPtr &>(tensor_) = new_tensor;
    if (stub_ != nullptr) {
      stub_->SetValue(new_tensor);
    }
    return new_tensor;
  }
  return tensor;
}

BaseTensorPtr TensorPy::GetBaseTensor() const {
  if (tensor_ == nullptr) {
    return std::static_pointer_cast<BaseTensor>(stub_->WaitValue());
  }
  return tensor_;
}

void TensorPy::UpdateStub(const BaseTensorPtr &tensor) { stub_->SetValue(tensor); }

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

void TensorPy::SetShape(const ShapeVector &shape) { GetBaseTensor()->set_shape(shape); }

bool TensorPy::IsPersistentData() const { return GetTensor()->is_persistent_data(); }

int TensorPy::DataDim() const { return GetBaseTensor()->DataDim(); }

TensorPy &TensorPy::AssignValue(const TensorPy &tensorpy) {
  // todo: assign value for base tensor.
  auto tensor = tensorpy.GetTensor();
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

std::string TensorPy::ToString() const {
  DataSync(true);
  return GetBaseTensor()->ToStringRepr();
}

std::string TensorPy::ToStringRepr() const { return GetBaseTensor()->ToStringRepr(); }

bool TensorPy::CheckStub() { return Tensor::CheckStub(); }

ParamInfoPtr TensorPy::GetParamInfo() const { return GetBaseTensor()->param_info(); }

void TensorPy::SetParamInfo(const ParamInfoPtr &param_info) {
  auto base_tensor = GetBaseTensor();
  MS_EXCEPTION_IF_NULL(base_tensor);
  base_tensor->set_param_info(param_info);
}

py::object TensorPy::GetFlattenTensor() { return flatten_tensor_; }

void TensorPy::SetFlattenTensor(py::object tensor) {
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
  TensorPyPtrList out;
  return out;
}

TensorPyPtrList TensorPy::GetFlattenedTensors(const TensorPyPtrList &tensorpys) {
  TensorPyPtrList result_tensorpys;
  return result_tensorpys;
}

bool TensorPy::IsComplex() const {
  auto base_tensor = GetBaseTensor();
  TypeId type_id = base_tensor->data_type();
  switch (type_id) {
    case TypeId::kNumberTypeComplex:
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex128:
      return true;
    default:
      break;
  }
  return false;
}

bool TensorPy::IsSigned() const {
  auto base_tensor = GetBaseTensor();
  TypeId type_id = base_tensor->data_type();
  switch (type_id) {
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt8:
    case TypeId::kNumberTypeInt16:
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt64:
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat64:
    case TypeId::kNumberTypeBFloat16:
    case TypeId::kNumberTypeComplex:
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex128:
      return true;
    default:
      break;
  }
  return false;
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

const py::object TensorPy::GetGrad() const {
  if (!grad_.check() || grad_.is_none()) {
    return py::none();
  }
  return grad_;
}

void TensorPy::SetGrad(const py::object &grad) { grad_ = grad; }

const py::object TensorPy::GetGradFn() const {
  if (!grad_fn_.check() || grad_fn_.is_none()) {
    return py::none();
  }
  return grad_fn_;
}

void TensorPy::SetGradFn(const py::object &grad_fn) { grad_fn_ = grad_fn; }

const py::object TensorPy::GetRequiresGrad() const {
  if (!requires_grad_.check() || requires_grad_.is_none()) {
    return py::none();
  }
  return requires_grad_;
}

void TensorPy::SetRequiresGrad(const py::object &requires_grad) { requires_grad_ = requires_grad; }

const py::object TensorPy::GetRetainGrad() const {
  if (!retain_grad_.check() || retain_grad_.is_none()) {
    return py::none();
  }
  return retain_grad_;
}

void TensorPy::SetRetainGrad(const py::object &retain_grad) { retain_grad_ = retain_grad; }

const py::object TensorPy::GetSliceNumOfPersistentData() const {
  if (!slice_num_of_persistent_data_.check() || slice_num_of_persistent_data_.is_none()) {
    return py::none();
  }
  return slice_num_of_persistent_data_;
}

void TensorPy::SetSliceNumOfPersistentData(const py::object &slice_num_of_persistent_data) {
  slice_num_of_persistent_data_ = slice_num_of_persistent_data;
}

const py::object TensorPy::GetSliceShapeOfPersistentData() const {
  if (!slice_shape_of_persistent_data_.check() || slice_shape_of_persistent_data_.is_none()) {
    return py::none();
  }
  return slice_shape_of_persistent_data_;
}

void TensorPy::SetSliceShapeOfPersistentData(const py::object &slice_shape_of_persistent_data) {
  slice_shape_of_persistent_data_ = slice_shape_of_persistent_data;
}

/* =========================================== Common Function ================================================= */
bool IsTensorPy(const py::handle &obj) {
  if (TensorPy_Type == nullptr || !obj.check()) {
    return false;
  }
  PyObject *raw_ptr = obj.ptr();
  PyObject *str_type = reinterpret_cast<PyObject *>(TensorPy_Type);
  return PyObject_IsInstance(raw_ptr, str_type);
}

const TensorPtr ConvertToTensor(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  PyObject *str_type = reinterpret_cast<PyObject *>(TensorPy_Type);
  if (PyObject_IsInstance(raw_ptr, str_type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    TensorPtr tensor_ptr = tensor->value.GetTensor();
    return tensor_ptr;
  }

  return nullptr;
}

py::object GetPythonTensor() {
  auto tensor_module = PyObjManager::Get().GetTensorModule();
  return py::reinterpret_borrow<py::object>(tensor_module);
}

const ValuePtr ConvertToValue(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  PyObject *str_type = reinterpret_cast<PyObject *>(TensorPy_Type);
  if (PyObject_IsInstance(raw_ptr, str_type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    auto &value = tensor->value;
    if (value.has_stub()) {
      return value.stub();
    }
    return value.GetBaseTensor();
  }
  MS_LOG(EXCEPTION) << "Not TensorPy object";
}

BaseTensorPtr ConvertToBaseTensor(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  PyObject *str_type = reinterpret_cast<PyObject *>(TensorPy_Type);
  if (PyObject_IsInstance(raw_ptr, str_type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    auto tensor_ptr = tensor->value.GetBaseTensor();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return tensor_ptr;
  }
  return nullptr;
}

PyType<TensorPy> *ConvertPyObject2TensorPyType(const py::object obj) {
  PyType<TensorPy> *tensor_type = reinterpret_cast<PyType<TensorPy> *>(obj.ptr());

  return tensor_type;
}

const py::handle ConvertToTensorPy(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  PyObject *str_type = reinterpret_cast<PyObject *>(TensorPy_Type);

  if (PyObject_IsInstance(raw_ptr, str_type)) {
    return obj;
  }

  return nullptr;
}

PyObject *TensorPythonInit(BaseTensorPtr tensor) {
  PyObject *tensorPythonClass = PyObject_GetAttrString(PyObjManager::Get().GetTensorModule(), "Tensor");
  PyObject *obj = (reinterpret_cast<PyTypeObject *>(tensorPythonClass))
                    ->tp_alloc(reinterpret_cast<PyTypeObject *>(tensorPythonClass), 0);
  if (obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  auto result = (PyType<TensorPy> *)obj;
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }

  new (&result->value) TensorPy(tensor);
  result->value.SetInitFinished(true);

  return reinterpret_cast<PyObject *>(result);
}

PyObject *TensorPythonInitFromTensor(TensorPtr tensor) {
  PyType<TensorPy> *result = (PyType<TensorPy> *)TensorPy_Type->tp_alloc(TensorPy_Type, 0);
  if (result == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }

  new (&result->value) TensorPy(tensor);
  // set to adapt python __repr__
  result->value.SetInitFinished(true);

  return reinterpret_cast<PyObject *>(result);
}

PyTypeObject *GetTensorPyType() { return TensorPy_Type; }

void SetTensorPyType(PyTypeObject *TensorPyType) { TensorPy_Type = TensorPyType; }

py::object PackTensorToPyObject(BaseTensorPtr tensor) {
  PyObject *tensor_py = TensorPythonInit(tensor);
  return py::reinterpret_steal<py::object>(tensor_py);
}

PyObject *PackTensor(const BaseTensorPtr &tensor) {
  PyObject *python_tensor_class = PyObject_GetAttrString(PyObjManager::Get().GetTensorModule(), "Tensor");
  auto tensor_py_type = reinterpret_cast<PyTypeObject *>(python_tensor_class);
  PyObject *obj = tensor_py_type->tp_alloc(tensor_py_type, 0);
  if (obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  auto result = (PyType<TensorPy> *)obj;
  new (&result->value) TensorPy(tensor);
  result->value.SetInitFinished(true);
  return reinterpret_cast<PyObject *>(result);
}

PyObject *Wrap(const BaseTensorPtr &tensor) { return PackTensor(tensor); }

PyObject *Wrap(const TensorPtr &tensor) { return PackTensor(tensor); }

PyObject *Wrap(const std::vector<BaseTensorPtr> &tensors) {
  PyObject *output = PyTuple_New(static_cast<Py_ssize_t>(tensors.size()));
  for (size_t i = 0; i < tensors.size(); ++i) {
    PyTuple_SET_ITEM(output, i, Wrap(tensors[i]));
  }
  return output;
}

PyObject *Wrap(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    return Wrap(value->cast<tensor::BaseTensorPtr>());
  } else if (value->isa<ValueSequeue>()) {
    auto sequeue = value->cast<ValueSequencePtr>();
    const auto &values = sequeue->value();
    size_t size = values.size();
    bool is_tuple = value->isa<ValueTuple>();
    PyObject *output =
      is_tuple ? PyTuple_New(static_cast<Py_ssize_t>(size)) : PyList_New(static_cast<Py_ssize_t>(size));
    for (size_t i = 0; i < size; ++i) {
      if (is_tuple) {
        PyTuple_SET_ITEM(output, i, Wrap(values[i]));
      } else {
        PyList_SET_ITEM(output, i, Wrap(values[i]));
      }
    }
    return output;
  } else {
    return ValueToPyData(value).release().ptr();
  }
}

PyTypeObject *getTensorPyType() { return TensorPy_Type; }

void setTensorPyType(PyTypeObject *TensorPyType) { TensorPy_Type = TensorPyType; }
}  // namespace tensor
}  // namespace mindspore
