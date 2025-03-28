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

#include <complex>
#include <algorithm>

#include "pybind11/complex.h"
#include "frontend/ir/tensor_py.h"
#include "include/common/pybind_api/api_register.h"
#include "abstract/abstract_value.h"
#include "pybind_api/ir/tensor_index_py.h"
#include "debug/profiler/profiler.h"
#include "include/backend/mbuf_device_address.h"
#include "utils/ordered_set.h"
#include "utils/ms_context.h"
#include "pybind_api/ir/tensor_register/tensor_func_reg.h"
#include "pybind_api/ir/tensor_register/auto_generate/tensor_py_gen.h"
#include "include/common/pynative/adapter.h"
#include "include/common/utils/exception.h"
#include "include/common/utils/pyobj_manager.h"

namespace mindspore {
namespace tensor {
PyTypeObject *TensorPyType = GetTensorPyType();
struct PyObjDeleter {
  void operator()(PyObject *object) const { Py_DECREF(object); }
};
using PyObjectPtr = std::unique_ptr<PyObject, PyObjDeleter>;
PyObjectPtr SafePtr1(PyObject *object) { return PyObjectPtr(object); }

// add for tensorpy
static PyObject *TensorPython_get_shape(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::tuple shape_tuple = obj->value.GetPyTupleShape();
  return shape_tuple.release().ptr();  // change to PyObject*
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_getShape(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  ShapeVector shape_tuple = obj->value.GetShape();
  return py::cast(shape_tuple).release().ptr();  // change to PyObject*
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_shape(PyObject *self, PyObject *list_obj, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  std::vector<int64_t> shape;
  if (!PyList_Check(list_obj)) {
    PyErr_SetString(PyExc_TypeError, "Expected a Python list.");
    return -1;
  }

  Py_ssize_t list_size = PyList_Size(list_obj);
  for (Py_ssize_t i = 0; i < list_size; ++i) {
    PyObject *item = PyList_GetItem(list_obj, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "List items must be integers.");
      shape.clear();
      return -1;
    }
    int64_t value = PyLong_AsLongLong(item);
    if (value == -1 && PyErr_Occurred()) {
      shape.clear();
      return -1;
    }
    shape.push_back(value);
  }

  obj->value.GetBaseTensor()->set_shape(shape);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

// setter
static PyObject *TensorPython_get_InitFinish(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  return PyBool_FromLong(obj->value.IsInitFinished() ? 1 : 0);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_ConstArg(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  return py::bool_(obj->value.IsConstArg()).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_init(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object init_obj = obj->value.GetInitializer();
  return init_obj.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_device(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  std::string deviceString = obj->value.GetDevice();
  return PyUnicode_FromString(deviceString.c_str());
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_ParentTensor(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object parentTensor_obj = obj->value.GetParentTensor();
  return parentTensor_obj.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_IndexOfParent(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object indexOfParent_obj = obj->value.GetIndexOfParent();
  return indexOfParent_obj.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_init_flag(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  return PyBool_FromLong(obj->value.GetTensor()->is_init() ? 1 : 0);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_adapter_flag(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  return PyBool_FromLong(obj->value.GetTensor()->is_adapter() ? 1 : 0);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_dtype(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  TypePtr type_ptr = obj->value.GetBaseTensor()->Dtype();
  return py::cast(type_ptr).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_size(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  size_t size = obj->value.GetBaseTensor()->DataSize();
  return PyLong_FromSize_t(size);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_itemsize(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  ssize_t itemsize = obj->value.GetBaseTensor()->data().itemsize();
  return PyLong_FromSsize_t(itemsize);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_nbytes(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  ssize_t nbytes = obj->value.GetBaseTensor()->data().nbytes();  // use data().nbytes()
  return PyLong_FromSsize_t(nbytes);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_strides(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  auto py_strides = TensorPybind::GetPyTupleStrides(*(obj->value.GetTensor()));
  return py_strides.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_paramInfo(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);

  return py::cast(obj->value.GetParamInfo()).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_Virtual(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  return py::bool_(obj->value.IsVirtual()).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_SymbolicShape(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object symbolicShape = obj->value.GetSymbolicShape();
  return symbolicShape.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_ConstArg(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  if (!PyBool_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The init_flag property value must be a boolean.");
    return -1;
  }
  obj->value.SetConstArg(value == Py_True);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_init(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object initializer_object = py::reinterpret_borrow<py::object>(value);
  obj->value.SetInitializer(initializer_object);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_device(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  const char *deviceString = PyUnicode_AsUTF8(value);
  obj->value.SetDevice(std::string(deviceString));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_ParentTensor(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetParentTensor(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_IndexOfParent(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetIndexOfParent(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_init_flag(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  if (!PyBool_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The init_flag property value must be a boolean.");
    return -1;
  }
  obj->value.GetTensor()->set_init_flag(value == Py_True);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_adapter_flag(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  if (!PyBool_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The adapter_flag property value must be a boolean.");
    return -1;
  }
  obj->value.GetTensor()->set_adapter_flag(value == Py_True);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_paramInfo(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  ParamInfoPtr paramInfo_object = py::cast<ParamInfoPtr>(value);
  obj->value.SetParamInfo(paramInfo_object);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_dtypeObj(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  TypePtr dtype_object = py::cast<TypePtr>(value);
  obj->value.SetDtype(dtype_object);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static int TensorPython_set_VirtualFlag(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  if (!PyBool_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The init_flag property value must be a boolean.");
    return -1;
  }
  obj->value.SetVirtualFlag(value == Py_True);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_slice_num_of_persistent_data_(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetSliceNumOfPersistentData();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_slice_num_of_persistent_data_(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetSliceNumOfPersistentData(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_slice_shape_of_persistent_data_(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetSliceShapeOfPersistentData();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_slice_shape_of_persistent_data_(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetSliceShapeOfPersistentData(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_grad(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetGrad();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_grad(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetGrad(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_grad_fn(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetGradFn();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_grad_fn(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetGradFn(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_requires_grad(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetRequiresGrad();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_requires_grad(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetRequiresGrad(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_get_retain_grad(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  py::object sliceNumOfPersistentData = obj->value.GetRetainGrad();
  return sliceNumOfPersistentData.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static int TensorPython_set_retain_grad(PyObject *self, PyObject *value, void *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  obj->value.SetRetainGrad(py::reinterpret_borrow<py::object>(value));
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyGetSetDef PyTensorPython_getseters[] = {
  {"param_info", (getter)TensorPython_get_paramInfo, (setter)TensorPython_set_paramInfo, "paramInfo of the tensor",
   nullptr},
  {"init_finished", (getter)TensorPython_get_InitFinish, nullptr,
   "Indicates whether the Tensor initialization is finished.", NULL},
  {"const_arg", (getter)TensorPython_get_ConstArg, (setter)TensorPython_set_ConstArg,
   "Whether the tensor is a constant when it is used for the argument of a network.", NULL},
  {"init", (getter)TensorPython_get_init, (setter)TensorPython_set_init, "The information of init data.", NULL},
  {"device", (getter)TensorPython_get_device, (setter)TensorPython_set_device,
   "This parameter is reserved and does not need to be configured.", NULL},
  {"parent_tensor_", (getter)TensorPython_get_ParentTensor, (setter)TensorPython_set_ParentTensor,
   "If current Tensor is an index value of another Tensor, set to another Tensor.", NULL},
  {"index_of_parent_", (getter)TensorPython_get_IndexOfParent, (setter)TensorPython_set_IndexOfParent,
   "index_of_parent_ will set to the index.", NULL},
  {"adapter_flag", (getter)TensorPython_get_adapter_flag, (setter)TensorPython_set_adapter_flag, "Get the adapter flag",
   NULL},
  {"init_flag", (getter)TensorPython_get_init_flag, (setter)TensorPython_set_init_flag, "Get the initialization flag",
   NULL},
  {"_dtype", (getter)TensorPython_get_dtype, (setter)TensorPython_set_dtypeObj, R"mydelimiter(
                                Get the tensor's data type.
   
                                Returns:
                                    type, the data type of tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                    >>> data.dtype
                                    Int32
                                )mydelimiter",
   NULL},
  {"dtype", (getter)TensorPython_get_dtype, nullptr, "Get the MetaTensor's dtype.", NULL},
  {"_shape", (getter)TensorPython_get_shape, (setter)TensorPython_set_shape, "Shape of the tensor", NULL},
  {"shape", (getter)TensorPython_getShape, nullptr, "Get the MetaTensor's shape.", NULL},
  {"_size", (getter)TensorPython_get_size, nullptr, R"mydelimiter(
                                Get tensor's data size.
   
                                Returns:
                                    size_t, the size of tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data.size
                                    6
                                )mydelimiter",
   nullptr},
  {"_itemsize", (getter)TensorPython_get_itemsize, nullptr, R"mydelimiter(
                                Get the tensor's length of one element in bytes.
   
                                Returns:
                                    itemsize, length of one element in bytes.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                    >>> data.itemsize
                                    4
                                )mydelimiter",
   nullptr},
  {"_nbytes", (getter)TensorPython_get_nbytes, nullptr, R"mydelimiter(
                                Get the tensor's total number of bytes.
   
                                Returns:
                                    nbytes, total number of bytes taken by the tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                    >>> data.nbytes
                                    4
                                )mydelimiter",
   nullptr},
  {"_strides", (getter)TensorPython_get_strides, nullptr, R"mydelimiter(
                                Get the tensor's tuple of bytes to step in each dimension
                                when traversing an array.
   
                                Returns:
                                    tuple[int], the strides of the tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                    >>> data.strides
                                    (4, 4)
                                )mydelimiter",
   nullptr},
  {"virtual_flag", (getter)TensorPython_get_Virtual, (setter)TensorPython_set_VirtualFlag, "Get the _virtual flag.",
   NULL},
  {"symbolic_shape", (getter)TensorPython_get_SymbolicShape, nullptr, "Get the symbolic shape.", NULL},
  {"slice_num_of_persistent_data_", (getter)TensorPython_get_slice_num_of_persistent_data_,
   TensorPython_set_slice_num_of_persistent_data_, "Get the slice_num_of_persistent_data_.", NULL},
  {"slice_shape_of_persistent_data_", (getter)TensorPython_get_slice_shape_of_persistent_data_,
   TensorPython_set_slice_shape_of_persistent_data_, "Get the slice_shape_of_persistent_data_.", NULL},
  {"_grad", (getter)TensorPython_get_grad, TensorPython_set_grad, "Get the _grad.", NULL},
  {"_grad_fn", (getter)TensorPython_get_grad_fn, TensorPython_set_grad_fn, "Get the _grad_fn.", NULL},
  {"_requires_grad", (getter)TensorPython_get_requires_grad, TensorPython_set_requires_grad, "Get the _requires_grad.",
   NULL},
  {"_retain_grad", (getter)TensorPython_get_retain_grad, TensorPython_set_retain_grad, "Get the _retain_grad.", NULL},
  {NULL}  // Sentinel
};

PyObject *TensorPy_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  PyType<TensorPy> *self;

  // alloc memory
  self = (PyType<TensorPy> *)type->tp_alloc(type, 0);
  return reinterpret_cast<PyObject *>(self);
}

static int TensorPy_pyinit(PyObject *obj, PyObject *args, PyObject *kwargs) {
  PyType<TensorPy> *self = reinterpret_cast<PyType<TensorPy> *>(obj);
  // parameter need to stop
  if (self->value.IsInitFinished()) {
    return 0;
  }
  struct TensorInitialization {
    PyObject *input_data_;
    PyObject *dtype_;
    PyObject *shape_;
    PyObject *init_;
    PyObject *const_arg_;
    PyObject *device_;
  };
  static const char *kws[] = {"input_data", "dtype", "shape", "init", "const_arg", "device", nullptr};
  constexpr const char fmt[] = "|OOOOOO:Tensor";
  TensorInitialization argsT = {Py_None, Py_None, Py_None, Py_None, Py_False, Py_None};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, fmt, const_cast<char **>(kws), &argsT.input_data_, &argsT.dtype_,
                                   &argsT.shape_, &argsT.init_, &argsT.const_arg_, &argsT.device_)) {
    MS_EXCEPTION(TypeError) << "Not support tensor input parameter type!!!";
  }
  py::dict p;
  HANDLE_MS_EXCEPTION
  p = GetPythonTensor().attr("_init")(
    py::cast<py::object>(py::handle(argsT.input_data_)), py::cast<py::object>(py::handle(argsT.dtype_)),
    py::cast<py::object>(py::handle(argsT.shape_)), py::cast<py::object>(py::handle(argsT.init_)),
    py::cast<py::object>(py::handle(argsT.const_arg_)), py::cast<py::object>(py::handle(argsT.device_)));
  TensorPtr tensor = TensorPyImpl::InitTensor(p);
  new (&self->value) TensorPy(tensor);
  self->value.SetInitializer(TensorPyImpl::GetInitializerFromPython(p));
  self->value.SetConstArg(TensorPyImpl::GetConstArgFromPython(p));
  self->value.SetDevice(TensorPyImpl::GetDeviceFromPython(p));
  self->value.SetSymbolicShape(TensorPyImpl::GetSymbolicShapeFromPython(p));
  self->value.SetInitFinished(true);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *TensorPython_set_paramInfo_(PyObject *, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *self;
  PyObject *value;
  if (!PyArg_ParseTuple(args, "OO", &self, &value)) {
    return nullptr;
  }
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  ParamInfoPtr paramInfo_object = py::cast<ParamInfoPtr>(value);
  obj->value.SetParamInfo(paramInfo_object);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_asnumpy(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *py_tensor;
  pybind11::array np_array;
  if (self == NULL) {
    PyObject *oriTensor;
    if (!PyArg_ParseTuple(args, "O", &oriTensor)) {
      return nullptr;
    }
    py_tensor = (PyType<TensorPy> *)oriTensor;
  } else {
    py_tensor = (PyType<TensorPy> *)self;
  }
  TensorPy &tensorPy = py_tensor->value;
  auto tensor = tensorPy.GetTensor();
  np_array = TensorPybind::SyncAsNumpy(*tensor);
  return np_array.release().ptr();  // usr ptr() to get PyObject*
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_data_sync(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  MS_LOG(DEBUG) << "Tensor data_sync";
  PyType<TensorPy> *py_tensor;
  bool need_wait;
  if (self != NULL) {
    py_tensor = (PyType<TensorPy> *)self;
    TensorPy &tensor = py_tensor->value;
    if (!PyArg_ParseTuple(args, "p", &need_wait)) {
      return nullptr;
    }
    tensor.DataSync(need_wait);
  } else {
    PyObject *oriTensor;
    if (!PyArg_ParseTuple(args, "Op", &oriTensor, &need_wait)) {
      return nullptr;
    }
    py_tensor = (PyType<TensorPy> *)oriTensor;
    TensorPy &tensor = py_tensor->value;
    tensor.DataSync(need_wait);
  }
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_repr(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *py_tensor = (PyType<TensorPy> *)self;
  std::string repr = py_tensor->value.ToStringRepr();
  return PyUnicode_FromString(repr.c_str());
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_from_numpy(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *numpy_array;
  if (!PyArg_ParseTuple(args, "O", &numpy_array)) {
    return nullptr;
  }
  if (!py::isinstance<py::array>(numpy_array)) {
    PyErr_SetString(PyExc_TypeError, "Expected a NumPy array.");
    return nullptr;
  }
  py::array input = py::cast<py::array>(numpy_array);
  return tensor::PackTensor(TensorPybind::MakeTensorOfNumpy(input));
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorIndex_setitem_index_info(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_data;
  PyObject *py_index;
  PyObject *py_value;
  int is_ascend = 1;
  if (!PyArg_ParseTuple(args, "OOO|i", &py_data, &py_index, &py_value, &is_ascend)) {
    return nullptr;
  }
  pybind11::object data = pybind11::reinterpret_borrow<pybind11::object>(py_data);
  pybind11::object index = pybind11::reinterpret_borrow<pybind11::object>(py_index);
  pybind11::object value = pybind11::reinterpret_borrow<pybind11::object>(py_value);
  pybind11::bool_ ascend = static_cast<bool>(is_ascend);
  py::object result;
  result = TensorIndex::SetItemIndexInfo(data, index, value, ascend);
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorIndex_getitem_index_info(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_data;
  PyObject *py_index;
  int is_ascend = 1;
  if (!PyArg_ParseTuple(args, "OO|i", &py_data, &py_index, &is_ascend)) {
    return nullptr;
  }
  py::object data = py::reinterpret_borrow<py::object>(py_data);
  py::object index = py::reinterpret_borrow<py::object>(py_index);
  py::bool_ ascend = static_cast<bool>(is_ascend);
  py::object result;

  result = TensorIndex::GetItemIndexInfo(data, index, ascend);
  if (result.is_none()) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorIndex_is_flattened(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *tensor_list;
  if (!PyArg_ParseTuple(args, "O", &tensor_list)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list of TensorPy objects.");
    return nullptr;
  }

  if (!PyList_Check(tensor_list)) {
    PyErr_SetString(PyExc_TypeError, "Argument must be a list.");
    return nullptr;
  }
  TensorPtrList tensors;
  for (Py_ssize_t i = 0; i < PyList_Size(tensor_list); i++) {
    PyObject *item = PyList_GetItem(tensor_list, i);
    PyType<TensorPy> *tensor_py_type = reinterpret_cast<PyType<TensorPy> *>(item);
    tensors.emplace_back(tensor_py_type->value.GetTensor());
  }
  bool result = Tensor::IsFlattened(tensors);
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_check_stub(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  bool result = Tensor::CheckStub();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPy_make_persistent_data_tensor(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_input;
  PyObject *py_slice_num;
  if (!PyArg_ParseTuple(args, "OO", &py_input, &py_slice_num)) {
    return nullptr;
  }
  py::array input = py::reinterpret_borrow<py::array>(py_input);
  py::int_ slice_num = py::reinterpret_borrow<py::int_>(py_slice_num);
  TensorPtr tensor = TensorPybind::MakePersistentDataTensorOfNumpy(input, slice_num);
  PyType<TensorPy> *result = (PyType<TensorPy> *)TensorPyType->tp_alloc(TensorPyType, 0);
  if (result == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  new (&result->value) TensorPy(tensor);
  result->value.SetInitFinished(true);
  return reinterpret_cast<PyObject *>(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_bytes(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  auto tensor = obj->value.GetTensor();
  py::bytes bytes = TensorPybind::GetBytes(*tensor);
  return bytes.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPy_convert_bytes_to_tensor(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *bytes_obj;     // py::bytes object
  PyObject *dims_obj;      // py::tuple object
  PyObject *type_ptr_obj;  // TypePtr object
  if (!PyArg_ParseTuple(args, "OOO", &bytes_obj, &dims_obj, &type_ptr_obj)) {
    return nullptr;
  }
  py::bytes bytes = py::reinterpret_borrow<py::bytes>(bytes_obj);
  py::tuple dims = py::reinterpret_borrow<py::tuple>(dims_obj);
  TypePtr type_ptr = py::cast<TypePtr>(py::handle(type_ptr_obj));
  TensorPyPtr tensor = TensorPyImpl::ConvertBytesToTensor(bytes, dims, type_ptr);
  PyType<TensorPy> *py_tensor = (PyType<TensorPy> *)TensorPyType->tp_alloc(TensorPyType, 0);
  if (py_tensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  new (&py_tensor->value) TensorPy(tensor->GetTensor());
  py_tensor->value.SetInitFinished(true);
  return reinterpret_cast<PyObject *>(py_tensor);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_flush_from_cache(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  auto tensor = obj->value.GetTensor();
  TensorPybind::FlushFromCache(*tensor);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_is_persistent_data(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *obj = reinterpret_cast<PyType<TensorPy> *>(self);
  bool result = obj->value.IsPersistentData();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_asnumpy_of_slice_persistent_data(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  int32_t param_key;
  int slice_index;
  if (!PyArg_ParseTuple(args, "ii", &param_key, &slice_index)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(self);
  auto tensorTmp = tensor->value.GetTensor();
  py::array np_array = TensorPybind::AsNumpyOfSlice(*tensorTmp, param_key, slice_index);
  return np_array.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_is_init(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(self);
  bool result = tensor->value.IsInit();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_set_initFlag(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(self);
  int flag;
  if (!PyArg_ParseTuple(args, "i", &flag)) {
    return nullptr;
  }
  tensor->value.SetInitFlag(static_cast<bool>(flag));
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_data_dim(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(self);
  int dim = tensor->value.DataDim();
  return PyLong_FromLong(dim);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_assign_value(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_tensor;
  if (!PyArg_ParseTuple(args, "O", &py_tensor)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  if (PyObject_TypeCheck(py_tensor, TensorPyType)) {
    PyType<TensorPy> *tensorpy = (PyType<TensorPy> *)(py_tensor);
    tensor->value.AssignValue(tensorpy->value);
  } else if (py::isinstance<mindspore::tensor::BaseTensor>(py_tensor) || py::isinstance<Tensor>(py_tensor)) {
    Tensor *tensor_data = reinterpret_cast<Tensor *>(py_tensor);
    tensor->value.GetTensor()->AssignValue(*tensor_data);
  } else {
    PyType<TensorPy> *tensorpy = (PyType<TensorPy> *)(py_tensor);
    tensor->value.AssignValue(tensorpy->value);
  }
  Py_INCREF(self);
  return self;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_set_dtype(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_type;

  if (!PyArg_ParseTuple(args, "O", &py_type)) {
    return nullptr;
  }
  TypePtr type_ptr = py::cast<TypePtr>(py::handle(py_type));
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  TypePtr result = tensor->value.SetDtype(type_ptr);

  return py::cast(result).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_offload(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  const char *file_path;
  if (!PyArg_ParseTuple(args, "s", &file_path)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool success = tensor->value.Offload(file_path);
  return PyBool_FromLong(success);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_offload_file_path(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  const std::string &file_path = tensor->value.GetOffloadFilePath();
  return PyUnicode_FromString(file_path.c_str());
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_move_to(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  const char *to;
  int blocking;
  if (!PyArg_ParseTuple(args, "si", &to, &blocking)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  auto tensorTmp = tensor->value.GetTensor();
  TensorPtr result = TensorPybind::MoveTo(*tensorTmp, std::string(to), blocking);
  TensorPyPtr tmpTensor = std::make_shared<TensorPy>(result);
  PyType<TensorPy> *resultTensor = (PyType<TensorPy> *)TensorPyType->tp_alloc(TensorPyType, 0);
  new (&resultTensor->value) TensorPy(tmpTensor->GetTensor());
  resultTensor->value.SetInitFinished(true);
  return reinterpret_cast<PyObject *>(resultTensor);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPy_set_user_data(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  const char *key;
  PyObject *value_obj;
  if (!PyArg_ParseTuple(args, "sO", &key, &value_obj)) {
    return nullptr;
  }
  TensorPybind::SetUserData(tensor->value.GetBaseTensor(), py::str(key), py::reinterpret_borrow<py::object>(value_obj));
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPy_get_user_data(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  const char *key;
  if (!PyArg_ParseTuple(args, "s", &key)) {
    return nullptr;
  }
  py::object result = TensorPybind::GetUserData(tensor->value.GetBaseTensor(), py::str(key));
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_set_cast_dtype(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyObject *dtype_obj = nullptr;
  if (!PyArg_ParseTuple(args, "|O", &dtype_obj)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  TypePtr dtype = nullptr;
  if (dtype_obj != nullptr) {
    dtype = py::cast<TypePtr>(py::handle(dtype_obj));
  }
  tensor->value.SetCastDtype(dtype);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_execute_lazy_task(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(self);
  tensor->value.ExecuteLazyTask();
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_is_contiguous(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool result = tensor->value.IsContiguous();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_is_complex(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool result = tensor->value.IsComplex();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_is_signed(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool result = tensor->value.IsSigned();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_stride(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  std::vector<int64_t> stride = tensor->value.GetStride();
  PyObject *py_stride = PyList_New(stride.size());
  for (size_t i = 0; i < stride.size(); ++i) {
    PyList_SetItem(py_stride, i, PyLong_FromLongLong(stride[i]));
  }

  return py_stride;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_get_storage_offset(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  int64_t offset = tensor->value.GetStorageOffset();
  return PyLong_FromLongLong(offset);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *RegisterTensorBackwardHook(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *tensor_obj;
  PyObject *hook_func;
  if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &hook_func)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)tensor_obj;
  py::function hook = py::cast<py::function>(hook_func);
  uint64_t hook_id = pynative::HookAdapter::RegisterTensorBackwardHook(tensor->value.GetBaseTensor(), hook);
  return PyLong_FromUnsignedLongLong(hook_id);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *RemoveTensorBackwardHook(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  uint64_t handle_id;
  if (!PyArg_ParseTuple(args, "K", &handle_id)) {
    return nullptr;
  }
  pynative::HookAdapter::RemoveTensorBackwardHook(handle_id);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_ToString(PyObject *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  std::string result = tensor->value.ToString();
  return PyUnicode_FromString(result.c_str());
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_SetOffload(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *tensor_obj;
  PyObject *releaseObj;
  if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &releaseObj)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)tensor_obj;
  bool release = (PyObject_IsTrue(releaseObj) == 1);
  auto tensorTmp = tensor->value.GetBaseTensor();
  TensorPybind::Offload(tensorTmp, release);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_set_device_address(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  uintptr_t addr;
  ShapeVector shape;
  PyObject *shape_obj;
  PyObject *type_ptr_obj;

  if (!PyArg_ParseTuple(args, "KOO", &addr, &shape_obj, &type_ptr_obj)) {
    return nullptr;
  }
  if (PyTuple_Check(shape_obj)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(shape_obj); ++i) {
      PyObject *item = PyTuple_GET_ITEM(shape_obj, i);
      shape.push_back(PyLong_AsLong(item));
    }
  } else if (PyList_Check(shape_obj)) {
    for (Py_ssize_t i = 0; i < PyList_Size(shape_obj); ++i) {
      PyObject *item = PyList_GetItem(shape_obj, i);
      shape.push_back(PyLong_AsLong(item));
    }
  } else {
    return nullptr;
  }
  TypePtr type_ptr = py::cast<TypePtr>(py::handle(type_ptr_obj));
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  auto tensorTmp = tensor->value.GetBaseTensor();
  TensorPybind::SetDeviceAddress(tensorTmp, addr, shape, type_ptr);

  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static py::object TensorGetItemImpl(const py::object &self, const py::object &py_index) {
  static std::string config_static_shape = common::GetEnv("MS_PYNATIVE_CONFIG_STATIC_SHAPE");
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice ||
      MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode || config_static_shape == "1") {
    return self.attr("_getitem_origin")(py_index);
  }
  return self.attr("_getitem")(py_index);
}

static PyObject *TensorPython_GetItem(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_index = NULL;
  if (!PyArg_ParseTuple(args, "O", &py_index)) {
    return nullptr;
  }
  py::object result =
    TensorGetItemImpl(py::reinterpret_borrow<py::object>(self), py::reinterpret_borrow<py::object>(py_index));
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static py::object TensorSetItemImpl(const py::object &self, const py::object &py_index, const py::object &py_value) {
  static std::string config_static_shape = common::GetEnv("MS_PYNATIVE_CONFIG_STATIC_SHAPE");
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice ||
      MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode || config_static_shape == "1") {
    return self.attr("_setitem_origin")(py_index, py_value);
  }
  return self.attr("_setitem")(py_index, py_value);
}

static PyObject *TensorPython_SetItem(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *py_index = NULL, *py_value = NULL;
  if (!PyArg_ParseTuple(args, "|OO", &py_index, &py_value)) {
    return nullptr;
  }
  py::object self_obj = py::reinterpret_borrow<py::object>(self);
  py::object py_index_obj = py::reinterpret_borrow<py::object>(py_index);
  py::object py_value_obj = py::reinterpret_borrow<py::object>(py_value);
  py::object result = TensorSetItemImpl(self_obj, py_index_obj, py_value_obj);
  if (result.is(py::none())) {
    return nullptr;
  }
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_GetFlattenTensors(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyObject *py_tensor_list;

  if (!PyArg_ParseTuple(args, "O", &py_tensor_list)) {
    return nullptr;
  }

  TensorPtrList tensors;
  std::vector<TensorPy> tensorpys;
  Py_ssize_t len = PyList_Size(py_tensor_list);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyList_GetItem(py_tensor_list, i);
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)item;
    tensors.push_back(tensor->value.GetTensor());
    tensorpys.push_back(tensor->value);
  }
  TensorPtrList out_tensors = Tensor::GetFlattenedTensors(tensors);
  if (out_tensors.empty()) {
    Py_RETURN_NONE;
  }
  std::map<TypeId, OrderedSet<PyObject *>> chunk_map;
  for (auto &tensorpy : tensorpys) {
    auto owner_tensorpy = tensorpy.GetFlattenTensor();
    auto get_normalize_type = [](TypeId id) {
      if (id == kNumberTypeFloat) {
        // kNumberTypeFloat is an alias of kNumberTypeFloat32.
        return kNumberTypeFloat32;
      }
      return id;
    };
    auto chunk_dtype = get_normalize_type(tensorpy.GetDataType());
    chunk_map[chunk_dtype].add(owner_tensorpy);
  }
  py::list result_tensorpys;
  for (const auto &entry : chunk_map) {
    const auto &chunk_tensors = entry.second;
    for (const auto &tensor_obj : chunk_tensors) {
      result_tensorpys.append(tensor_obj);
    }
  }
  return result_tensorpys.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_FlattenTensors(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyObject *py_tensor_list = NULL;
  PyObject *fusion_size_ori = NULL;
  size_t fusion_size = 0;
  static const char *kwlist[] = {"py_tensor_list", "fusion_size", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", const_cast<char **>(kwlist), &py_tensor_list,
                                   &fusion_size_ori)) {
    return nullptr;
  }
  if (fusion_size_ori != NULL) {
    fusion_size = (size_t)PyLong_AsUnsignedLongLong(fusion_size_ori);
  }
  TensorPtrList tensors;
  std::vector<TensorPy> tensorpys;
  Py_ssize_t len = PyList_Size(py_tensor_list);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyList_GetItem(py_tensor_list, i);
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)item;
    tensors.push_back(tensor->value.GetTensor());
    tensorpys.push_back(tensor->value);
  }
  TensorPtrList out_tensors = Tensor::FlattenTensors(tensors, fusion_size);
  PyObject *resultList = PyList_New(out_tensors.size());
  int index = 0;
  for (auto &tensor : out_tensors) {
    TensorPyPtr tmpTensor = std::make_shared<TensorPy>(tensor);
    PyType<TensorPy> *result = (PyType<TensorPy> *)TensorPyType->tp_alloc(TensorPyType, 0);
    new (&result->value) TensorPy(tmpTensor->GetTensor());
    result->value.SetInitFinished(true);
    PyObject *outTensor = reinterpret_cast<PyObject *>(result);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject *item = PyList_GetItem(py_tensor_list, i);
      PyType<TensorPy> *tensorPython = (PyType<TensorPy> *)item;
      auto flatten = Tensor::GetFlattenedTensor(tensorPython->value.GetTensor());
      if (tensor == flatten) {
        // need to store falttened python tensor
        Py_INCREF(outTensor);
        tensorPython->value.SetFlattenTensor(outTensor);
      }
    }
    PyList_SetItem(resultList, index++, outTensor);
  }
  return resultList;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_GetFusionSize(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyObject *py_tensor_list;

  if (!PyArg_ParseTuple(args, "O", &py_tensor_list)) {
    return nullptr;
  }

  TensorPtrList tensorpys;
  Py_ssize_t len = PyList_Size(py_tensor_list);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyList_GetItem(py_tensor_list, i);
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)item;
    tensorpys.push_back(tensor->value.GetTensor());
  }
  size_t out = Tensor::GetFusionSize(tensorpys);
  return PyLong_FromSize_t(out);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_GetNewItem(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  auto tensorTmp = tensor->value.GetBaseTensor();
  py::object result = TensorPybind::Item(tensorTmp);
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_ToList(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  auto tensorTmp = tensor->value.GetBaseTensor();
  py::object result = TensorPybind::ToList(tensorTmp);
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_HasAutoGrad(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool result = tensor->value.HasAutoGrad();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_GetHooks(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  auto tensorTmp = tensor->value.GetBaseTensor();
  py::list result = pynative::HookAdapter::GetHooks(tensorTmp);
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_GetDataPtr(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyObject *tensor_obj;
  if (self != nullptr) {
    tensor_obj = self;
  } else if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
    return nullptr;
  }
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)tensor_obj;
  auto tensorTmp = tensor->value.GetBaseTensor();
  uintptr_t dataPtr = TensorPybind::DataPtr(tensorTmp);
  PyObject *dataPtrResult = PyLong_FromVoidPtr(reinterpret_cast<void *>(dataPtr));
  return dataPtrResult;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_NeedContiguous(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)self;
  bool result = tensor->value.NeedContiguous();
  return PyBool_FromLong(result);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_SetLoad(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *obj;
  if (self != nullptr) {
    obj = self;
  } else if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }
  PyType<TensorPy> *tensorObj = (PyType<TensorPy> *)obj;
  auto tensor = tensorObj->value.GetTensor();
  TensorPybind::Load(*tensor);
  Py_RETURN_NONE;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_getstate(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *state;
  if (self != nullptr) {
    state = self;
  } else {
    if (!PyArg_ParseTuple(args, "O", &state)) {
      return nullptr;
    }
  }

  PyType<TensorPy> *tensor = (PyType<TensorPy> *)state;
  auto tensorTmp = tensor->value.GetTensor();
  py::array numpy_array = TensorPybind::SyncAsNumpy(*tensorTmp);
  py::tuple result = py::make_tuple(numpy_array);
  return result.release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *TensorPython_setstate(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  PyObject *state;
  PyObject *tensor;
  if (self != nullptr) {
    tensor = self;
    if (!PyArg_ParseTuple(args, "O", &state)) {
      return nullptr;
    }
  } else if (!PyArg_ParseTuple(args, "OO", &tensor, &state)) {
    return nullptr;
  }
  if (!PyTuple_Check(state) || PyTuple_Size(state) != 1) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid state!");
    return nullptr;
  }
  py::tuple t = py::reinterpret_borrow<py::tuple>(state);
  py::dict p;
  p["input_data"] = t[0].cast<py::array>();
  PyType<TensorPy> *resultTensor = (PyType<TensorPy> *)tensor;
  if (resultTensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  TensorPtr tensorPy = TensorPyImpl::InitTensor(p);
  new (&resultTensor->value) TensorPy(tensorPy);
  resultTensor->value.SetInitializer(TensorPyImpl::GetInitializerFromPython(p));
  resultTensor->value.SetConstArg(TensorPyImpl::GetConstArgFromPython(p));
  resultTensor->value.SetDevice(TensorPyImpl::GetDeviceFromPython(p));
  resultTensor->value.SetSymbolicShape(TensorPyImpl::GetSymbolicShapeFromPython(p));
  resultTensor->value.SetInitFinished(true);
  Py_INCREF(resultTensor);
  return reinterpret_cast<PyObject *>(resultTensor);
  HANDLE_MS_EXCEPTION_END
}

static PyMethodDef Tensor_methods[] = {
  {"set_param_info", (PyCFunction)TensorPython_set_paramInfo_, METH_STATIC | METH_VARARGS, "set param info"},
  {"asnumpy", (PyCFunction)TensorPython_asnumpy, METH_VARARGS, R"mydelimiter(
                                Convert tensor to numpy.ndarray.
   
                                Returns:
                                    numpy.ndarray.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> array = data.asnumpy()
                                    >>> array
                                    array([[1., 1., 1.],
                                           [1., 1., 1.]])
                                )mydelimiter"},
  {"data_sync", (PyCFunction)TensorPython_data_sync, METH_VARARGS, "Synchronize data with optional wait"},
  {"__repr__", (PyCFunction)TensorPython_repr, METH_NOARGS, "Return the string representation of the tensor."},
  {"from_numpy", TensorPython_from_numpy, METH_STATIC | METH_VARARGS, R"mydelimiter(
                                Creates a Tensor from a numpy.ndarray without copy.
   
                                Arg:
                                    array (numpy.ndarray): The input ndarray.
   
                                Returns:
                                    Tensor, tensor with shared data to input ndarray.
   
                                Examples:
                                    >>> a = np.ones((2, 3))
                                    >>> t = mindspore.Tensor.from_numpy(a)
                                )mydelimiter"},
  {"setitem_index_info", TensorIndex_setitem_index_info, METH_STATIC | METH_VARARGS, "Set item index information."},
  {"getitem_index_info", TensorIndex_getitem_index_info, METH_STATIC | METH_VARARGS, "Get item index information."},
  {"_is_flattened", TensorIndex_is_flattened, METH_STATIC | METH_VARARGS, "Index is flattened."},
  {"_is_test_stub", TensorPython_check_stub, METH_STATIC | METH_NOARGS, "Check if this is a test stub."},
  {"persistent_data_from_numpy", TensorPy_make_persistent_data_tensor, METH_VARARGS,
   R"mydelimiter(
                                Creates a Tensor from a numpy.ndarray without copy.
                                Use persistent data tensor.
   
                                Arg:
                                    array (numpy.ndarray): The input ndarray.
                                    slice_num (int): The slice num of persistent data tensor.
   
                                Returns:
                                    Tensor, tensor with shared data to input ndarray.
   
                                Examples:
                                    >>> a = np.ones((2, 3))
                                    >>> t = mindspore.Tensor.persistent_data_from_numpy(a, 1)
                                )mydelimiter"},
  {"get_bytes", TensorPython_get_bytes, METH_VARARGS, R"mydelimiter(
                                Get raw data of tensor with type of bytes.
   
                                Returns:
                                    Bytes of tensor.
   
                                Examples:
                                    >>> import mindspore as ms
                                    >>> from mindspore import Tensor
                                    >>> x = ms.Tensor([1, 2, 3], ms.int16)
                                    >>> print(x.get_bytes())
                                    b'\x01\x00\x02\x00\x03\x00'
                                )mydelimiter"},
  {"convert_bytes_to_tensor", TensorPy_convert_bytes_to_tensor, METH_STATIC | METH_VARARGS,
   R"mydelimiter(
                                Convert raw data to tensor.
   
                                Returns:
                                    Tensor.
   
                                Examples:
                                    >>> import mindspore as ms
                                    >>> from mindspore import Tensor
                                    >>> x = Tensor([1, 2, 3], ms.int16)
                                    >>> out = Tensor.convert_bytes_to_tensor(x.get_bytes(), x.shape, x.dtype)
                                    >>> print(x.asnumpy())
                                    [1 2 3]
                                )mydelimiter"},
  {"_flush_from_cache", TensorPython_flush_from_cache, METH_NOARGS, R"mydelimiter(
                                Flush Cache data to Host if tensor is cache enable.
   
                                Returns:
                                    None.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data._flush_from_cache()
                                )mydelimiter"},
  {"is_persistent_data", TensorPython_is_persistent_data, METH_NOARGS, R"mydelimiter(
                                Check if tensor have persistent data.
   
                                Returns:
                                    Bool.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data.is_persistent_data()
                                )mydelimiter"},
  {"asnumpy_of_slice_persistent_data", TensorPython_asnumpy_of_slice_persistent_data, METH_VARARGS,
   R"mydelimiter(
                                Convert tensor to numpy.ndarray of a slice.
   
                                Returns:
                                    numpy.ndarray.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2000000000, 256)))
                                    >>> data.asnumpy_of_slice_persistent_data(0, 1)
                                )mydelimiter"},
  {"is_init", TensorPython_is_init, METH_NOARGS, R"mydelimiter(
                                Get tensor init_flag.
   
                                Returns:
                                    bool, whether the tensor init.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data.is_init()
                                    False
                                )mydelimiter"},
  {"set_init_flag", TensorPython_set_initFlag, METH_VARARGS, R"mydelimiter(
                                Set tensor init_flag.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data.set_init_flag(True)
                                )mydelimiter"},
  {"dim", TensorPython_data_dim, METH_VARARGS, R"mydelimiter(
                                Get tensor's data dimension.
   
                                Returns:
                                    int, the dimension of tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((2, 3)))
                                    >>> data.dim()
                                    2
                                )mydelimiter"},
  {"assign_value_cpp", TensorPython_assign_value, METH_VARARGS, R"mydelimiter(
                                Assign another tensor value to this.
   
                                Arg:
                                    value (:class:`mindspore.tensor`): The value tensor.
   
                                Examples:
                                    >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                    >>> data2 = mindspore.Tensor(np.ones((2, 2), np.float32))
                                    >>> data.assign_value(data2)
                                    >>> data.shape
                                    (2, 2)
                                )mydelimiter"},
  {"set_dtype", TensorPython_set_dtype, METH_VARARGS, R"mydelimiter(
                                 Set the tensor's data type.
   
                                 Arg:
                                     dtype (:class:`mindspore.dtype`): The type of output tensor.
   
                                 Examples:
                                     >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                     >>> data.set_dtype(mindspore.int32)
                                     mindspore.int32
                                 )mydelimiter"},
  {"offload", TensorPython_offload, METH_VARARGS, R"mydelimiter(
                                 Offload tensor data to file.
   
                                 Arg:
                                     str : file path to save tensor data.
                                 Returns:
                                     bool, whether the tensor offload success.
                                 Examples:
                                     >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                     >>> data.offload('./test.data')
                                     True
                                 )mydelimiter"},
  {"offload_file_path", TensorPython_get_offload_file_path, METH_NOARGS, R"mydelimiter(
                                 Offload file path for tensor.
   
                                 Returns:
                                    str, offload file path for tensor.
                                 Examples:
                                     >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                     >>> ret = data.offload('./test.data')
                                     >>> ret = (data.offload_file_path() != '')
                                     True
                                 )mydelimiter"},
  {"move_to", TensorPython_move_to, METH_VARARGS, R"mydelimiter(
                                  Copy tensor between host and device asynchronously if blocking=False,
                                  otherwise synchronously. if the arg `to`=`CPU`, means D2H copy;
                                  if the arg `to`=`GPU` or `to`=`ASCEND`, means H2D copy.
   
                                  Args:
                                      str: A string, "CPU" or "ASCEND" or "GPU".
                                      bool: A bool type value, Default: ``True`` .
   
                                  Returns:
                                         Tensor, with the same type and shape as the "self".
   
                                 Examples:
                                     >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                     >>> ret = data.move_to("CPU")
                                 )mydelimiter"},
  {"_set_user_data", TensorPy_set_user_data, METH_VARARGS, "Set user data for a tensor."},
  {"_get_user_data", TensorPy_get_user_data, METH_VARARGS, "Get user data for a tensor."},
  {"set_cast_dtype", (PyCFunction)TensorPython_set_cast_dtype, METH_VARARGS | METH_KEYWORDS,
   "Set the casting data type for the tensor."},
  {"wait_pipeline", (PyCFunction)TensorPython_execute_lazy_task, METH_NOARGS,
   "Execute pending tasks in the tensor pipeline."},
  {"is_contiguous", (PyCFunction)TensorPython_is_contiguous, METH_NOARGS,
   "Check if the tensor is contiguous in memory."},
  {"is_complex", (PyCFunction)TensorPython_is_complex, METH_NOARGS, R"mydelimiter(
                              Check if the type of tensor is complex.
                              For details, please refer to: func:'mindspore.ops.is_complex'.

                              Returns:
                                  A Boolean, check whether the tensor is of complex type.

                              Examples
                                  >>> x = mindspore.Tensor([1 + 1j], dtype=mindspore.complex128)
                                  >>> y = mindspore.Tensor([1], dtype=mindspore.int32)
                                  >>> x.is_complex()
                                  Ture
                                  >>> y.is_complex()
                                  False
                              )mydelimiter"},
  {"is_signed", (PyCFunction)TensorPython_is_signed, METH_NOARGS, R"mydelimiter(
                              Judge whether the data type of tensor is a signed data type.

                              Returns:
                                  A Boolean, check whether the tensor is of signed type.

                              Examples:
                                  >>> x = mindspore.Tensor([1.0], dtype=mindspore.float32)
                                  >>> y = mindspore.Tensor([1], dtype=mindspore.uint32)
                                  >>> x.is_signed()
                                  Ture
                                  >>> y.is_signed()
                                  False
                              )mydelimiter"},
  {"stride", (PyCFunction)TensorPython_get_stride, METH_NOARGS, "Get the stride of the tensor."},
  {"storage_offset", (PyCFunction)TensorPython_get_storage_offset, METH_NOARGS,
   "Get the storage offset of the tensor."},
  {"register_hook", (PyCFunction)RegisterTensorBackwardHook, METH_STATIC | METH_VARARGS,
   "Register a backward hook for a tensor."},
  {"remove_hook", (PyCFunction)RemoveTensorBackwardHook, METH_STATIC | METH_VARARGS,
   "Remove a backward hook for a tensor using its handle ID."},
  {"__str__", (PyCFunction)TensorPython_ToString, METH_NOARGS, "Return a string representation of the tensor."},
  {"_offload", (PyCFunction)TensorPython_SetOffload, METH_STATIC | METH_VARARGS, "Set offload for the tensor."},
  {"set_device_address", TensorPython_set_device_address, METH_VARARGS, "Set the device address for the tensor."},
  {"__getitem__", (PyCFunction)TensorPython_GetItem, METH_VARARGS, "Get item from TensorPy"},
  {"__setitem__", (PyCFunction)TensorPython_SetItem, METH_VARARGS, "Set item to TensorPy"},
  {"__getstate__", (PyCFunction)TensorPython_getstate, METH_VARARGS, "Get the state of the TensorPy object"},
  {"__setstate__", (PyCFunction)TensorPython_setstate, METH_VARARGS, "Set the state of the TensorPy object"},
  {"_get_flattened_tensors", (PyCFunction)TensorPython_GetFlattenTensors, METH_STATIC | METH_VARARGS,
   "Flatten the input tensors."},
  {"_flatten_tensors", (PyCFunction)TensorPython_FlattenTensors, METH_STATIC | METH_VARARGS | METH_KEYWORDS,
   "Flatten the input tensors."},
  {"_get_fusion_size", (PyCFunction)TensorPython_GetFusionSize, METH_STATIC | METH_VARARGS,
   "Flatten the input tensors."},
  {"_item", (PyCFunction)TensorPython_GetNewItem, METH_VARARGS | METH_KEYWORDS, R"mydelimiter(
                               Return the value of this tensor as standard Python number.
                               This only works for tensors with one element.
   
                               Returns:
                                   A scalar, type is defined by the dtype of the Tensor.
   
                               Examples:
                                   # index is None:
                                   >>> t = mindspore.Tensor([1])
                                   >>> t.item()
                                   1
                               )mydelimiter"},
  {"_tolist", (PyCFunction)TensorPython_ToList, METH_VARARGS, R"mydelimiter(
                                Convert a Tensor to List. If the input is Tensor scalar, a Python scalar will be returned.
    
                                Returns:
                                    List or Python scalar.
    
                                Examples:
                                    >>> x = ms.Tensor([[1, 2, 3], [4, 5, 6]])
                                    >>> out1 = x.tolist()
                                    >>> print(out1)
                                    [[1, 2, 3], [4, 5, 6]]
                                    >>> out2 = x[0][0].tolist()
                                    >>> print(out2)
                                    1
                                )mydelimiter"},
  {"_has_auto_grad", (PyCFunction)TensorPython_HasAutoGrad, METH_VARARGS | METH_KEYWORDS, "HasAutoGrad."},
  {"hooks", (PyCFunction)TensorPython_GetHooks, METH_VARARGS | METH_KEYWORDS, "get hooks."},
  {"_data_ptr", (PyCFunction)TensorPython_GetDataPtr, METH_VARARGS, "get Data ptr."},
  {"_need_contiguous", (PyCFunction)TensorPython_NeedContiguous, METH_VARARGS | METH_KEYWORDS, "need Contiguous."},
  {"_load", (PyCFunction)TensorPython_SetLoad, METH_VARARGS, "SetLoad."},
  {NULL, NULL, 0, NULL}};

static void TensorPy_pydealloc(PyObject *obj) {
  PyType<TensorPy> *self = reinterpret_cast<PyType<TensorPy> *>(obj);
  // Init tensor failed and don't need to exec ~TensorPy.
  if (self->value.IsInitFinished()) {
    self->value.~TensorPy();
  } else {
    MS_LOG(WARNING) << "The tensor has not complete initialization and no need to execute destructor.";
  }
  // release Python self
  Py_TYPE(obj)->tp_free(obj);
}

void RegPyTensorMethods() {
  int total_size = 0;
  for (auto &arr : {Tensor_methods, TensorMethods}) {
    for (int i = 0; arr[i].ml_name != NULL; i++) total_size++;
  }

  // alloc
  PyMethodDef *merged = new PyMethodDef[total_size + 1];  // +1 use to set end
  int idx = 0;
  for (auto &arr : {Tensor_methods, TensorMethods}) {
    for (int i = 0; arr[i].ml_name != NULL; i++) {
      merged[idx++] = arr[i];
    }
  }
  merged[idx] = {NULL, NULL, 0, NULL};  // end tig
  TensorPyType->tp_methods = merged;
}
void RegPyTensor(py::module *m) {
  m->def("addStubTensorMethods", &RegStubTensorMethods, "add StubTensorMethods");
  PyHeapTypeObject *heap_type = reinterpret_cast<PyHeapTypeObject *>(PyType_Type.tp_alloc(&PyType_Type, 0));
  if (!heap_type) {
    MS_LOG(ERROR) << "heap_type is null";
    return;
  }

  static const char *type_name = "TensorPy";
  PyObjectPtr name = SafePtr1(PyUnicode_FromString(type_name));
  PyObjectPtr qualname = SafePtr1(PyUnicode_FromString(type_name));

  heap_type->ht_name = name.release();
  heap_type->ht_qualname = qualname.release();
  TensorPyType = &heap_type->ht_type;
  TensorPyType->tp_name = type_name;
  TensorPyType->tp_basicsize = sizeof(PyType<TensorPy>);
  TensorPyType->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  TensorPyType->tp_new = TensorPy_pynew;
  TensorPyType->tp_init = TensorPy_pyinit;
  TensorPyType->tp_getset = PyTensorPython_getseters;
  TensorPyType->tp_dealloc = TensorPy_pydealloc;
  RegPyTensorMethods();
  if (PyType_Ready(TensorPyType) < 0) {
    MS_LOG(ERROR) << "TensorPyType ready < 0";
    return;
  }
  // set __module__
  PyObject *module_name = PyUnicode_FromString("mindspore._c_expression");
  if (module_name == NULL) {
    return;
  }
  if (PyObject_SetAttrString(reinterpret_cast<PyObject *>(TensorPyType), "__module__", module_name) < 0) {
    Py_DECREF(module_name);
    return;
  }
  Py_DECREF(module_name);
  PyObject *abc_meta = PyObjManager::Get().GetAbcModule();
  if (abc_meta == nullptr) {
    PyErr_Print();
    return;
  }
  PyObject *ABCMeta = PyObject_GetAttrString(abc_meta, "ABCMeta");
  if (ABCMeta == nullptr) {
    PyErr_Print();
    return;
  }
  if (PyObject_SetAttrString(reinterpret_cast<PyObject *>(TensorPyType), "__metaclass__", ABCMeta) < 0) {
    PyErr_Print();
    return;
  }
  SetTensorPyType(TensorPyType);
  TensorPyType = GetTensorPyType();
  m->add_object("TensorPy", reinterpret_cast<PyObject *>(TensorPyType));
}

void RegMetaTensor(const py::module *m) {
  // Define TensorData as a python class so that ownership of tensor data can be managed.
  (void)py::class_<TensorData, TensorDataPtr>(*m, "_TensorData");
}

void RegCSRTensor(const py::module *m) {
  // Define python CSRTensor class.
  (void)py::class_<CSRTensor, std::shared_ptr<CSRTensor>>(*m, "CSRTensor")
    .def(py::init(
           [](const py::object &indptr, const py::object &indices, const py::object &values, const py::tuple &shape) {
             return std::make_shared<CSRTensor>(ConvertToTensor(indptr), ConvertToTensor(indices),
                                                ConvertToTensor(values), TensorPyImpl::GetShapeFromTuple(shape));
           }),
         py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const py::object &csr_tensor) {
           auto csr_tensor_ = csr_tensor.cast<CSRTensorPtr>().get();
           return std::make_shared<CSRTensor>(*csr_tensor_);
         }),
         py::arg("input"))
    .def_property_readonly("_shape", CSRTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &CSRTensor::Dtype)
    .def_property_readonly("_indptr", CSRTensorPy::GetIndptr)
    .def_property_readonly("_indices", CSRTensorPy::GetIndices)
    .def_property_readonly("_values", CSRTensorPy::GetValues)
    .def("__str__", &CSRTensor::ToString)
    .def("__repr__", &CSRTensor::ToString);
}

void RegCOOTensor(const py::module *m) {
  // Define python COOTensor class.
  (void)py::class_<COOTensor, std::shared_ptr<COOTensor>>(*m, "COOTensor")
    .def(py::init([](const py::object &indices, const py::object &values, const py::tuple &shape) {
           return std::make_shared<COOTensor>(ConvertToTensor(indices), ConvertToTensor(values),
                                              TensorPyImpl::GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const py::object &coo_tensor) {
           auto coo_tensor_ = coo_tensor.cast<COOTensorPtr>().get();
           return std::make_shared<COOTensor>(*coo_tensor_);
         }),
         py::arg("input"))
    .def_property_readonly("_shape", COOTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &COOTensor::Dtype)
    .def_property_readonly("_indices", COOTensorPy::GetIndices)
    .def_property_readonly("_values", COOTensorPy::GetValues)
    .def("__str__", &COOTensor::ToString)
    .def("__repr__", &COOTensor::ToString);
}

void RegRowTensor(const py::module *m) {
  // Define python RowTensor class.
  (void)py::class_<RowTensor, std::shared_ptr<RowTensor>>(*m, "RowTensor")
    .def(py::init([](const py::object &indices, const py::object &values, const py::tuple &shape) {
           return std::make_shared<RowTensor>(ConvertToTensor(indices), ConvertToTensor(values),
                                              TensorPyImpl::GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const py::object &row_tensor) {
           auto row_tensor_ = row_tensor.cast<RowTensorPtr>().get();
           return std::make_shared<RowTensor>(*row_tensor_);
         }),
         py::arg("input"))
    .def_property_readonly("_shape", RowTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &RowTensor::Dtype)
    .def_property_readonly("_indices", RowTensorPy::GetIndices)
    .def_property_readonly("_values", RowTensorPy::GetValues)
    .def("__str__", &RowTensor::ToString)
    .def("__repr__", &RowTensor::ToString);
}
}  // namespace tensor
}  // namespace mindspore
