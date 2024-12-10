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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_

#include <memory>
#include <vector>
#include <functional>
#include <string>

#include "pybind11/pybind11.h"

#include "ir/tensor.h"
#include "include/common/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace tensor {
class COMMON_EXPORT TensorPy;
using TensorPyPtr = std::shared_ptr<TensorPy>;
using TensorPyPtrList = std::vector<std::shared_ptr<TensorPy>>;

class COMMON_EXPORT TensorPy {
 public:
  TensorPy() = default;
  explicit TensorPy(const TensorPy &input);
  explicit TensorPy(const BaseTensorPtr &input);
  explicit TensorPy(const TensorPtr &input);
  explicit TensorPy(int64_t input, const TypePtr &data_type = nullptr);
  explicit TensorPy(int32_t input, const TypePtr &data_type = nullptr);
  explicit TensorPy(int16_t input, const TypePtr &data_type = nullptr);
  explicit TensorPy(int8_t input, const TypePtr &data_type = nullptr);
  explicit TensorPy(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);
  explicit TensorPy(const std::vector<int32_t> &input, const TypePtr &data_type = nullptr);
  explicit TensorPy(const std::vector<double> &input, const TypePtr &data_type = nullptr);
  explicit TensorPy(const std::vector<float> &input, const TypePtr &data_type = nullptr);
  TensorPy(TypeId data_type, const ShapeVector &shape);
  ~TensorPy() = default;

  bool IsInitFinished();
  void SetInitFinished(bool flag);
  bool IsConstArg();
  void SetConstArg(bool flag);
  bool IsVirtual();
  void SetVirtualFlag(bool flag);
  const py::object GetInitializer() const;
  void SetInitializer(const py::object &init);
  const std::string GetDevice() const;
  void SetDevice(const std::string &dev);
  const TensorPtr GetTensor() const;
  const BaseTensorPtr GetBaseTensor() const;
  const py::object GetParentTensor();
  void SetParentTensor(const py::object &parent);
  const py::object GetIndexOfParent();
  void SetIndexOfParent(const py::object &index);
  py::tuple GetPyTupleShape();
  py::int_ GetPyItemSize();
  py::int_ GetPyNBytes();
  py::tuple GetPyTupleStrides();
  TypePtr GetDtype() const;
  TypePtr SetDtype(const TypePtr type);
  TypeId GetDataType() const;
  const ShapeVector &GetShape() const;
  bool IsInit() const;
  void SetInitFlag(bool flag);
  bool IsAdapter() const;
  void SetAdapterFlag(bool flag);
  void SetShape(const ShapeVector &shape);
  bool IsPersistentData() const;
  int DataDim() const;
  TensorPy &AssignValue(const TensorPy &tensorpy);
  bool Offload(const std::string &file_path);
  const std::string GetOffloadFilePath() const;
  void SetCastDtype(const TypePtr &dtype = nullptr);
  void DataSync(bool need_wait = true) const;
  void ExecuteLazyTask() const;
  bool IsContiguous() const;
  std::vector<int64_t> GetStride() const;
  const int64_t GetStorageOffset() const;
  std::string ToString() const;
  std::string ToStringRepr() const;
  static bool CheckStub();
  ParamInfoPtr GetParamInfo() const;
  void SetParamInfo(const ParamInfoPtr &param_info);
  const py::object GetSymbolicShape() const;
  void SetSymbolicShape(const py::object &symbolic);
  const size_t GetDataSize() const;
  void *GetTensorDataObject() const;
  const DeviceSyncPtr GetDeviceAddress() const;
  bool IsMSParameterOutput() const;
  void SetMSParameterOutput(bool flag);
  static TensorPyPtrList FlattenTensors(const TensorPyPtrList &tensorpys, size_t fusion_size = 0);
  static bool IsFlattened(const TensorPyPtrList &tensorpys);
  static TensorPyPtrList GetFlattenedTensors(const TensorPyPtrList &tensorpys);
  static size_t GetFusionSize(const TensorPyPtrList &flat_tensorpys);
  bool HasAutoGrad() const;
  bool NeedContiguous() const;

 private:
  bool init_finished_flag_{false};
  bool const_arg_flag_{false};
  bool virtual_flag_{false};
  bool ms_parameter_output_{false};
  py::object initializer_;
  py::object parent_tensor_;
  py::object index_of_parent_;
  py::object symbolic_shape_;
  std::string device_;
  BaseTensorPtr tensor_{nullptr};
  TensorPyPtr flatten_tensor_{nullptr};

  const TensorPyPtr GetFlattenTensor();
  void SetFlattenTensor(const TensorPyPtr tensor);
};

COMMON_EXPORT bool IsTensorPy(const py::handle &obj);
COMMON_EXPORT const TensorPyPtr ConvertToTensorPy(const py::handle &obj);
COMMON_EXPORT const TensorPtr ConvertToTensor(const py::handle &obj);

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_
