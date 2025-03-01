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
#include "include/common/profiler.h"
#include "include/backend/mbuf_device_address.h"
#include "utils/ordered_set.h"
#include "pybind_api/ir/tensor_register/tensor_func_reg.h"

namespace mindspore {
namespace tensor {
void RegTensorPyProperty(py::class_<TensorPy, std::shared_ptr<TensorPy>> *tensor_class) {
  tensor_class->def_property_readonly("init_finished", &TensorPy::IsInitFinished);
  tensor_class->def_property("const_arg", &TensorPy::IsConstArg, &TensorPy::SetConstArg);
  tensor_class->def_property("init", &TensorPy::GetInitializer, &TensorPy::SetInitializer);
  tensor_class->def_property("virtual_flag", &TensorPy::IsVirtual, &TensorPy::SetVirtualFlag);
  tensor_class->def_property("device", &TensorPy::GetDevice, &TensorPy::SetDevice);
  tensor_class->def_property("parent_tensor_", &TensorPy::GetParentTensor, &TensorPy::SetParentTensor);
  tensor_class->def_property("index_of_parent_", &TensorPy::GetIndexOfParent, &TensorPy::SetIndexOfParent);
  tensor_class->def_property_readonly("symbolic_shape", &TensorPy::GetSymbolicShape);
  tensor_class->def_property("__ms_parameter_output__", &TensorPy::IsMSParameterOutput,
                             &TensorPy::SetMSParameterOutput);

  tensor_class->def_property_readonly("dtype", &TensorPy::GetDtype, "Get the MetaTensor's dtype.");
  tensor_class->def_property_readonly("shape", &TensorPy::GetShape, "Get the MetaTensor's shape.");
  tensor_class->def_property("param_info", &TensorPy::GetParamInfo, &TensorPy::SetParamInfo);

  tensor_class->def_property("init_flag", &TensorPy::IsInit, &TensorPy::SetInitFlag);
  tensor_class->def_property("adapter_flag", &TensorPy::IsAdapter, &TensorPy::SetAdapterFlag);
  tensor_class->def_property("_dtype", &TensorPy::GetDtype, &TensorPy::SetDtype, R"mydelimiter(
                             Get the tensor's data type.

                             Returns:
                                 type, the data type of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.dtype
                                 Int32
                             )mydelimiter");
  tensor_class->def_property("_shape", &TensorPy::GetPyTupleShape, &TensorPy::SetShape);
  tensor_class->def_property_readonly("_size", &TensorPy::GetDataSize, R"mydelimiter(
                             Get tensor's data size.

                             Returns:
                                 size_t, the size of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.size
                                 6
                             )mydelimiter");
  tensor_class->def_property_readonly("_itemsize", &TensorPy::GetPyItemSize, R"mydelimiter(
                             Get the tensor's length of one element in bytes.

                             Returns:
                                 itemsize, length of one element in bytes.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.itemsize
                                 4
                             )mydelimiter");
  tensor_class->def_property_readonly("_nbytes", &TensorPy::GetPyNBytes, R"mydelimiter(
                             Get the tensor's total number of bytes.

                             Returns:
                                 nbytes, total number of bytes taken by the tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.nbytes
                                 4
                             )mydelimiter");
  tensor_class->def_property_readonly("_strides", &TensorPy::GetPyTupleStrides, R"mydelimiter(
                             Get the tensor's tuple of bytes to step in each dimension
                             when traversing an array.

                             Returns:
                                 tuple[int], the strides of the tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.strides
                                 (4, 4)
                             )mydelimiter");

  tensor_class->def_property("slice_num_of_persistent_data_", &TensorPy::GetSliceNumOfPersistentData,
                             &TensorPy::SetSliceNumOfPersistentData);
  tensor_class->def_property("slice_shape_of_persistent_data_", &TensorPy::GetSliceShapeOfPersistentData,
                             &TensorPy::SetSliceShapeOfPersistentData);
  tensor_class->def_property("_grad", &TensorPy::GetGrad, &TensorPy::SetGrad);
  tensor_class->def_property("_grad_fn", &TensorPy::GetGradFn, &TensorPy::SetGradFn);
  tensor_class->def_property("_requires_grad", &TensorPy::GetRequiresGrad, &TensorPy::SetRequiresGrad);
  tensor_class->def_property("_retain_grad", &TensorPy::GetRetainGrad, &TensorPy::SetRetainGrad);
}

void RegTensorPyFunction(py::class_<TensorPy, std::shared_ptr<TensorPy>> *tensor_class) {
  tensor_class->def("_flatten_tensors", TensorPy::FlattenTensors, py::arg("fusion_size") = 0);
  tensor_class->def("setitem_index_info", TensorIndex::SetItemIndexInfo);
  tensor_class->def("getitem_index_info", TensorIndex::GetItemIndexInfo);
  tensor_class->def("_is_flattened", TensorPy::IsFlattened);
  tensor_class->def("_get_flattened_tensors", TensorPy::GetFlattenedTensors);
  tensor_class->def("_get_fusion_size", TensorPy::GetFusionSize);
  tensor_class->def("_is_test_stub", TensorPy::CheckStub);
  tensor_class->def("from_numpy", TensorPyImpl::MakeTensorOfNumpy, R"mydelimiter(
                             Creates a Tensor from a numpy.ndarray without copy.

                             Arg:
                                 array (numpy.ndarray): The input ndarray.

                             Returns:
                                 Tensor, tensor with shared data to input ndarray.

                             Examples:
                                 >>> a = np.ones((2, 3))
                                 >>> t = mindspore.Tensor.from_numpy(a)
                             )mydelimiter");
  tensor_class->def("persistent_data_from_numpy", TensorPyImpl::MakePersistentDataTensorOfNumpy, R"mydelimiter(
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
                             )mydelimiter");
  tensor_class->def("get_bytes", TensorPyImpl::GetBytes, R"mydelimiter(
                             Get raw data of tensor with type of bytes.

                             Returns:
                                 Bytes of tensor.

                             Examples:
                                 >>> import mindspore as ms
                                 >>> from mindspore import Tensor
                                 >>> x = ms.Tensor([1, 2, 3], ms.int16)
                                 >>> print(x.get_bytes())
                                 b'\x01\x00\x02\x00\x03\x00'
                             )mydelimiter");
  tensor_class->def("convert_bytes_to_tensor", TensorPyImpl::ConvertBytesToTensor, R"mydelimiter(
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
                             )mydelimiter");
  tensor_class->def("asnumpy", TensorPyImpl::SyncAsNumpy, R"mydelimiter(
                             Convert tensor to numpy.ndarray.

                             Returns:
                                 numpy.ndarray.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> array = data.asnumpy()
                                 >>> array
                                 array([[1., 1., 1.],
                                        [1., 1., 1.]])
                             )mydelimiter");
  tensor_class->def("_flush_from_cache", TensorPyImpl::FlushFromCache, R"mydelimiter(
                             Flush Cache data to Host if tensor is cache enable.

                             Returns:
                                 None.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data._flush_from_cache()
                             )mydelimiter");
  tensor_class->def("is_persistent_data", &TensorPy::IsPersistentData, R"mydelimiter(
                             Check if tensor have persistent data.

                             Returns:
                                 Bool.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.is_persistent_data()
                             )mydelimiter");
  tensor_class->def("asnumpy_of_slice_persistent_data", TensorPyImpl::AsNumpyOfSlice, R"mydelimiter(
                             Convert tensor to numpy.ndarray of a slice.

                             Returns:
                                 numpy.ndarray.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2000000000, 256)))
                                 >>> data.asnumpy_of_slice_persistent_data(0, 1)
                             )mydelimiter");
  tensor_class->def("is_init", &TensorPy::IsInit, R"mydelimiter(
                             Get tensor init_flag.

                             Returns:
                                 bool, whether the tensor init.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.is_init()
                                 False
                             )mydelimiter");
  tensor_class->def("set_init_flag", &TensorPy::SetInitFlag, R"mydelimiter(
                             Set tensor init_flag.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.set_init_flag(True)
                             )mydelimiter");
  tensor_class->def("dim", &TensorPy::DataDim, R"mydelimiter(
                             Get tensor's data dimension.

                             Returns:
                                 int, the dimension of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.dim()
                                 2
                             )mydelimiter");
  tensor_class->def("assign_value_cpp", &TensorPy::AssignValue, R"mydelimiter(
                             Assign another tensor value to this.

                             Arg:
                                 value (:class:`mindspore.tensor`): The value tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                 >>> data2 = mindspore.Tensor(np.ones((2, 2), np.float32))
                                 >>> data.assign_value(data2)
                                 >>> data.shape
                                 (2, 2)
                             )mydelimiter");
  tensor_class->def("set_dtype", &TensorPy::SetDtype, R"mydelimiter(
                              Set the tensor's data type.

                              Arg:
                                  dtype (:class:`mindspore.dtype`): The type of output tensor.

                              Examples:
                                  >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                  >>> data.set_dtype(mindspore.int32)
                                  mindspore.int32
                              )mydelimiter");
  tensor_class->def("offload", &TensorPy::Offload, R"mydelimiter(
                              Offload tensor data to file.

                              Arg:
                                  str : file path to save tensor data.
                              Returns:
                                  bool, whether the tensor offload success.
                              Examples:
                                  >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                  >>> data.offload('./test.data')
                                  True
                              )mydelimiter");
  tensor_class->def("offload_file_path", &TensorPy::GetOffloadFilePath, R"mydelimiter(
                              Offload file path for tensor.

                              Returns:
                                 str, offload file path for tensor.
                              Examples:
                                  >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                  >>> ret = data.offload('./test.data')
                                  >>> ret = (data.offload_file_path() != '')
                                  True
                              )mydelimiter");
  tensor_class->def("move_to", TensorPyImpl::MoveTo, py::arg("to"), py::arg("blocking") = nullptr, R"mydelimiter(
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
                              )mydelimiter");
  tensor_class->def("_set_user_data", TensorPyImpl::SetUserData);
  tensor_class->def("_get_user_data", TensorPyImpl::GetUserData);
  tensor_class->def("set_cast_dtype", &TensorPy::SetCastDtype, py::arg("dtype") = nullptr);
  tensor_class->def("data_sync", &TensorPy::DataSync);
  tensor_class->def("wait_pipeline", &TensorPy::ExecuteLazyTask);
  tensor_class->def("is_contiguous", &TensorPy::IsContiguous);
  tensor_class->def("stride", &TensorPy::GetStride);
  tensor_class->def("storage_offset", &TensorPy::GetStorageOffset);
  tensor_class->def("register_hook", TensorPyImpl::RegisterTensorBackwardHook);
  tensor_class->def("remove_hook", TensorPyImpl::RemoveTensorBackwardHook);
  tensor_class->def("__str__", &TensorPy::ToString);
  tensor_class->def("__repr__", &TensorPy::ToStringRepr);
  tensor_class->def("_offload", TensorPyImpl::SetOffload);
  tensor_class->def("_offload", TensorPyImpl::SetOffload, py::arg("release"));
  tensor_class->def("_load", TensorPyImpl::SetLoad);
  tensor_class->def("set_device_address", TensorPyImpl::SetDeviceAddress, py::arg("addr"), py::arg("shape"),
                    py::arg("type_ptr"));
  tensor_class->def("__getitem__", TensorPybind::TensorGetItem);
  tensor_class->def("__setitem__", TensorPybind::TensorSetItem);
  tensor_class->def(py::pickle(
    [](const TensorPyPtr &t) {  // __getstate__
      /* Return a tuple that fully encodes the state of the object */
      return py::make_tuple(TensorPyImpl::SyncAsNumpy(t));
    },
    [](const py::tuple &t) {  // __setstate__
      if (t.size() != 1) {
        throw std::runtime_error("Invalid state!");
      }
      /* Create a new C++ instance */
      py::dict param;
      param["input_data"] = t[0].cast<py::array>();
      return TensorPyImpl::InitTensorPy(param);
    }));
  tensor_class->def("_item", TensorPyImpl::Item, R"mydelimiter(
                            Return the value of this tensor as standard Python number.
                            This only works for tensors with one element.

                            Returns:
                                A scalar, type is defined by the dtype of the Tensor.

                            Examples:
                                # index is None:
                                >>> t = mindspore.Tensor([1])
                                >>> t.item()
                                1
                            )mydelimiter");
  tensor_class->def("_tolist", TensorPyImpl::ToList, R"mydelimiter(
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
                            )mydelimiter");
  tensor_class->def("_has_auto_grad", &TensorPy::HasAutoGrad);
  tensor_class->def("hooks", TensorPyImpl::GetHooks);
  tensor_class->def("_data_ptr", TensorPyImpl::DataPtr);
  tensor_class->def("_need_contiguous", &TensorPy::NeedContiguous);
}

void RegTensorPy(const py::module *m) {
  auto tensorpyClass = py::class_<TensorPy, std::shared_ptr<TensorPy>>(*m, "TensorPy");
  tensorpyClass.def(py::init([](py::args &va, py::kwargs &kw) {
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
    TensorInitialization args = {Py_None, Py_None, Py_None, Py_None, Py_False, Py_None};
    if (!PyArg_ParseTupleAndKeywords(va.ptr(), kw.ptr(), fmt, const_cast<char **>(kws), &args.input_data_, &args.dtype_,
                                     &args.shape_, &args.init_, &args.const_arg_, &args.device_)) {
      MS_EXCEPTION(TypeError) << "Not support tensor input parameter type!!!";
    }
    auto p = TensorPyImpl::GetPythonTensor().attr("_init")(
      py::cast<py::object>(py::handle(args.input_data_)), py::cast<py::object>(py::handle(args.dtype_)),
      py::cast<py::object>(py::handle(args.shape_)), py::cast<py::object>(py::handle(args.init_)),
      py::cast<py::object>(py::handle(args.const_arg_)), py::cast<py::object>(py::handle(args.device_)));
    return TensorPyImpl::InitTensorPy(p);
  }));
  RegTensorPyProperty(&tensorpyClass);
  RegTensorPyFunction(&tensorpyClass);
  RegTensorFunc(&tensorpyClass);
}

void RegMetaTensor(const py::module *m) {
  // Define TensorData as a python class so that ownership of tensor data can be managed.
  (void)py::class_<TensorData, TensorDataPtr>(*m, "_TensorData");
}

void RegCSRTensor(const py::module *m) {
  // Define python CSRTensor class.
  (void)py::class_<CSRTensor, std::shared_ptr<CSRTensor>>(*m, "CSRTensor")
    .def(py::init([](const TensorPyPtr &indptr, const TensorPyPtr &indices, const TensorPyPtr &values,
                     const py::tuple &shape) {
           return std::make_shared<CSRTensor>(indptr->GetTensor(), indices->GetTensor(), values->GetTensor(),
                                              TensorPyImpl::GetShapeFromTuple(shape));
         }),
         py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const CSRTensor &csr_tensor) { return std::make_shared<CSRTensor>(csr_tensor); }),
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
    .def(py::init([](const TensorPyPtr &indices, const TensorPyPtr &values, const py::tuple &shape) {
           return std::make_shared<COOTensor>(indices->GetTensor(), values->GetTensor(),
                                              TensorPyImpl::GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const COOTensor &coo_tensor) { return std::make_shared<COOTensor>(coo_tensor); }),
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
    .def(py::init([](const TensorPyPtr &indices, const TensorPyPtr &values, const py::tuple &shape) {
           return std::make_shared<RowTensor>(indices->GetTensor(), values->GetTensor(),
                                              TensorPyImpl::GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const RowTensor &row_tensor) { return std::make_shared<RowTensor>(row_tensor); }),
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
