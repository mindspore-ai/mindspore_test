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

#ifndef MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
#define MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_

#include <memory>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "frontend/ir/dlpack_utils.h"

#include "ir/tensor.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/visible.h"
#include "frontend/np_dtypes/np_dtypes.h"

namespace py = pybind11;
namespace pybind11 {
namespace detail {
// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) const {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type)) {
      return false;
    }

    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }

    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});

    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<float16>() {}
};

template <>
struct type_caster<float16> : public npy_scalar_caster<float16> {
  static constexpr auto name = "float16";
};

template <>
struct npy_format_descriptor<bfloat16> {
  static constexpr auto name = "bfloat16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(mindspore::GetBFloat16NpDType());
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<bfloat16>() {}
};

template <>
struct type_caster<bfloat16> : public npy_scalar_caster<bfloat16> {
  static constexpr auto name = "bfloat16";
};

template <>
struct type_caster<mindspore::tensor::TensorPtr> {
  PYBIND11_TYPE_CASTER(mindspore::tensor::TensorPtr, _("Tensor"));
  bool load(handle src, bool) {
    if (mindspore::tensor::IsTensorPy(src)) {
      value = mindspore::tensor::ConvertToTensor(src);
      return true;
    }
    return false;
  }
  static handle cast(const mindspore::tensor::TensorPtr &src, return_value_policy, handle) {
    return handle(mindspore::tensor::Wrap(src));
  }
};
}  // namespace detail
}  // namespace pybind11

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of Mindsporeession project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {

// Tensor python wrapper and adapter class.
class FRONTEND_EXPORT TensorPybind {
 public:
  /// \brief Create Tensor from a numpy array object.
  ///
  /// \param[in] input [py::array] Data value of the tensor.
  /// \param[in] type_ptr [TypePtr] Data type of the tensor.
  static TensorPtr MakeTensor(const py::array &input, const TypePtr &type_ptr = nullptr);

  /// \brief Create Tensor from a numpy array without copy.
  ///
  /// \param[in] input [py::array] Data value of the tensor.
  static TensorPtr MakeTensorOfNumpy(const py::array &input);

  static bool IsPinned(const TensorPy &tensor);

  static TensorPtr MakePinMemoryTensor(const TensorPy &tensor);

  static py::bytes GetBytes(const Tensor &tensor);

  static py::buffer_info GetPyBufferFromPyArray(const py::array &input);

  static TensorPtr ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                        const TypePtr &type_ptr = nullptr);

  static py::object ToList(const TensorPtr &tensor);

  static py::object Item(const TensorPtr &tensor);

  static py::array SyncAsNumpy(const Tensor &tensor);

  static py::array NumpyNonBlocking(const Tensor &tensor);

  static py::array AsNumpy(const Tensor &tensor);

  static TensorPtr FromDLPack(const py::object &dlpack_capsule);

  static py::object ToDLPack(const py::object &tensor);
  static py::tuple GetPyTupleShape(const Tensor &tensor);

  static py::tuple GetPyTupleStrides(const Tensor &tensor);

  static py::int_ GetPyItemSize(const Tensor &tensor);

  static py::int_ GetPyNBytes(const Tensor &tensor);

  static void FlushFromCache(const Tensor &tensor);

  static void Offload(const TensorPtr &tensor, bool release);

  static void Load(const Tensor &tensor);

  static bool SharedMemory(const TensorPtr &tensor);

  // move tensor from device to host, or host to device asynchronously
  static TensorPtr MoveTo(const Tensor &self, const std::string &to, bool blocking = True);

  static void SetDeviceAddress(const TensorPtr &tensor, uintptr_t addr, const ShapeVector &shape,
                               const TypePtr type_ptr);

  static uintptr_t DataPtr(const TensorPtr &tensor);

  static std::string GetDevice(const TensorPtr &tensor);

  static void SetUserData(const TensorPtr &tensor, const py::str &key, const py::object &value);

  static py::object GetUserData(const TensorPtr &tensor, const py::str &key);
};

// CSRTensor python wrapper and adapter class.
class FRONTEND_EXPORT CSRTensorPy {
 public:
  static py::tuple GetPyTupleShape(const CSRTensor &csr_tensor);
  static py::object GetIndptr(const CSRTensorPtr &csr_tensor);
  static py::object GetIndices(const CSRTensorPtr &csr_tensor);
  static py::object GetValues(const CSRTensorPtr &csr_tensor);
};

// COOTensor python wrapper and adapter class.
class FRONTEND_EXPORT COOTensorPy {
 public:
  static py::tuple GetPyTupleShape(const COOTensor &coo_tensor);
  static py::object GetIndices(const COOTensorPtr &coo_tensor);
  static py::object GetValues(const COOTensorPtr &coo_tensor);
};

// RowTensor python wrapper and adapter class.
class FRONTEND_EXPORT RowTensorPy {
 public:
  static py::tuple GetPyTupleShape(const RowTensor &row_tensor);
  static py::object GetIndices(const RowTensorPtr &row_tensor);
  static py::object GetValues(const RowTensorPtr &row_tensor);
};

class FRONTEND_EXPORT TensorPyImpl {
 public:
  /// \brief Create a C++ Tensor.
  ///
  /// \param[in] input [py::dict] The input form python, as like: {"input_data": input_data, "dtype": dtype,
  /// "init": init, "const_arg": const_arg, "device": device, "symbolic_shape": symbolic_shape}.
  ///
  /// \return A C++ Tensor.
  static TensorPtr InitTensor(const py::dict &input);

  /// \brief Get the initialization form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The initialization.
  static py::object GetInitializerFromPython(const py::dict &input);

  /// \brief Get the constant argument form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The constant argument.
  static bool GetConstArgFromPython(const py::dict &input);

  /// \brief Get the device info form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The device info.
  static std::string GetDeviceFromPython(const py::dict &input);

  /// \brief Get the dynamically optimize shape form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The dynamically optimize shape.
  static py::object GetSymbolicShapeFromPython(const py::dict &input);

  /// \brief Get the type of Tensor form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ////
  /// \return The type of Tensor.
  static const TypePtr GetDtypeFromPython(const py::dict &input);

  /// \brief Get the shape of Tensor form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The shape of Tensor.
  static const ShapeVector GetShapeFromPython(const py::dict &input);

  /// \brief Create a TensorPy.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return A TensorPy.
  static const TensorPyPtr InitTensorPy(const py::dict &input);

  /// \brief  Create TensorPy from a numpy array without copy.
  ///
  /// \param[in] input [py::array] Data value of the tensorpy.
  ///
  /// \return This pointer address of TensorPy.
  static TensorPyPtr MakeTensorOfNumpy(const py::array &input);

  /// \brief Convert python object to Tensor.
  ///
  /// \param[in] bytes_obj [py::bytes] Python object.
  /// \param[in] dims [py::tuple] The dimensions.
  /// \param[in] type_ptr [TypePtr] The data type.
  ///
  /// \return A created TensorPy.
  static TensorPyPtr ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims, const TypePtr &type_ptr);

  /// \brief Release device address of graph output tensor by TensorPy.
  ///
  /// \param[in] tensorpy [TensorPyPtr] The TensorPy.
  /// \param[in] release [bool] Is release device address of graph output tensor.
  static void SetOffload(const TensorPyPtr &tensorpy, bool release);

  /// \brief Load device address of graph input tensor by TensorPy.
  ///
  /// \param[in] tensorpy [TensorPyPtr] The TensorPy.
  static void SetLoad(const TensorPyPtr &tensorpy);

  /// \brief Get Tensor data pointer for c++ type, and put it to py::bytes.
  ///
  /// \param[in] tensorpy [TensorPyPtr] The TensorPy.
  ///
  /// \return The pointer in the object.
  static py::bytes GetBytes(const TensorPyPtr &tensorpy);

  /// \brief Convert asynchronous Tensor into numpy data.
  ///
  /// \param[in] tensorpy [TensorPyPtr] The TensorPy.
  ///
  /// \return The numpy data.
  static py::array SyncAsNumpy(const TensorPyPtr &tensorpy);
  static void FlushFromCache(const TensorPyPtr &tensorpy);
  static TensorPyPtr FromDLPack(const py::object &dlpack_capsule);
  static py::object ToDLPack(const py::object &tensor);
  static TensorPyPtr MoveTo(const TensorPyPtr &tensorpy, const std::string &to, bool blocking = True);
  static void SetDeviceAddress(const TensorPyPtr &tensorpy, uintptr_t addr, const ShapeVector &shape,
                               const TypePtr type_ptr);
  static void SetUserData(const TensorPyPtr &tensorpy, const py::str &key, const py::object &value);
  static const py::object GetUserData(const TensorPyPtr &tensorpy, const py::str &key);
  static py::object ToList(const TensorPyPtr &tensorpy);
  static py::object Item(const TensorPyPtr &tensorpy);
  static uint64_t RegisterTensorBackwardHook(const TensorPyPtr &tensorpy, const py::function &hook);
  static void RemoveTensorBackwardHook(uint64_t handle_id);
  static py::list GetHooks(const TensorPyPtr &tensorpy);
  static uintptr_t DataPtr(const TensorPyPtr &tensorpy);
  static ShapeVector GetShapeFromTuple(const py::tuple &tuple);

 private:
  static TensorPtr InitTensorByInputDta(const py::dict &input, const TypePtr &dtype);
  static TensorPtr InitTensorByShape(const py::dict &input, const TypePtr &dtype);
};
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
