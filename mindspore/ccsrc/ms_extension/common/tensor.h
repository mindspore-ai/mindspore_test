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

#ifndef MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
#define MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
#include <set>
#include <memory>
#include <vector>
#include "pybind11/pybind11.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ms_extension/common/visible.h"

namespace mindspore {
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace tensor
class Value;
using ValuePtr = std::shared_ptr<Value>;
}  // namespace mindspore

namespace ms {
using TypeId = mindspore::TypeId;

/**
 * @class [API] Tensor
 * @brief Represents a tensor object in MindSpore, providing methods to manipulate and query its properties.
 */
class EXTENSION_API Tensor {
 public:
  /**
   * @brief [API] Constructs a placeholder Tensor.
   *
   * This default constructor creates an undefined Tensor, which acts as a placeholder.
   */
  Tensor() = default;

  /**
   * @brief [API] Constructs a Tensor with a specified data type and shape.
   *
   * This constructor initializes a Tensor object based on the given data type
   * and shape. The resulting Tensor will be allocated but uninitialized.
   *
   * @param type_id The data type of the Tensor.
   * @param shape The shape of the Tensor, represented as a vector of integers.
   */
  Tensor(TypeId type_id, const ShapeVector &shape);

  /**
   * @brief [API] Checks if the Tensor is defined.
   *        A defined Tensor has valid data and metadata, while an undefined Tensor does not.
   * @return True if the Tensor is defined, false otherwise.
   */
  bool is_defined() const { return _tensor_holder_ != nullptr; }

  /**
   * @brief [API] Retrieves the data type of the Tensor.
   * @return The data type of the Tensor.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  TypeId data_type() const;

  /**
   * @brief [API] Retrieves the shape of the Tensor.
   * @return A reference to the shape vector of the Tensor.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const ShapeVector &shape() const;

  /**
   * @brief [API] Returns the total number of elements in the tensor.
   * @return The total number of elements.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  size_t numel() const;

  /**
   * @brief [API] Calculates the stride of the Tensor.
   * @return A vector representing the strides of the Tensor along each dimension.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  std::vector<int64_t> stride() const;

  /**
   * @brief [API] Retrieves the storage offset of the Tensor.
   * @return The offset (in terms of elements) from the beginning of the storage.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  int64_t storage_offset() const;

  /**
   * @brief [API] Checks if the Tensor is stored contiguously in memory.
   * @return True if the Tensor is stored contiguously, false otherwise.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  bool is_contiguous() const;

  /**
   * @brief [API] Sets whether the Tensor needs contiguous memory.
   *        By default, `need_contiguous` is set to true. If non-contiguous storage is required,
   *        this method should be called with `false` before invoking `ms::pynative::PyboostRunner::Call`.
   * @param flag A boolean value indicating whether the Tensor should be stored contiguously.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  void SetNeedContiguous(bool flag) const;

  /**
   * @brief [API] Retrieves a pointer to the data of the Tensor.
   * @return A void pointer to the Tensor's raw data.
   * @throws If the Tensor is null, an exception is thrown.
   */
  void *GetDataPtr() const;

 public:
  /**
   * @brief Constructs a Tensor object from a given ValuePtr.
   * @param value A smart pointer to a MindSpore Value object. If the value is null, an undefined Tensor is constructed.
   *              Default nullptr.
   */
  explicit Tensor(const mindspore::ValuePtr &value);

  /**
   * @brief Deconstructor
   */
  ~Tensor() = default;

  /**
   * @brief Checks if the Tensor requires contiguous memory.
   * @return True if the Tensor needs to be stored contiguously, false otherwise.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  bool need_contiguous() const;

  /**
   * @brief Retrieves the stub node associated with the Tensor.
   * @return A smart pointer to the stub node (ValuePtr).
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const mindspore::ValuePtr &stub_node() const;

  /**
   * @brief Retrieves the underlying tensor object.
   * @return A smart pointer to the TensorPtr object.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const mindspore::tensor::TensorPtr &tensor() const;

  /**
   * @brief Converts the stub node to a Tensor object.
   *        This ensures that the Tensor is fully realized from its stub representation.
   *        After conversion, the stub node is released.
   */
  void ConvertStubNodeToTensor() const;

 private:
  /**
   * @struct RealTensorHolder
   * @brief Holds the actual data and metadata of the Tensor object.
   */
  struct EXTENSION_EXPORT RealTensorHolder {
    explicit RealTensorHolder(const mindspore::ValuePtr &value);

    // Indicates if the Tensor data needs to be contiguous. Defaults to true.
    bool need_contiguous_{true};
    // The value associated with the Tensor.
    mindspore::ValuePtr value_{nullptr};
    // The underlying Tensor object.
    mindspore::tensor::TensorPtr tensor_{nullptr};
  };

  // Shared pointer to the Tensor's holder.
  std::shared_ptr<RealTensorHolder> _tensor_holder_{nullptr};
};
}  // namespace ms

namespace pybind11 {
namespace detail {
template <>
struct EXTENSION_EXPORT type_caster<ms::Tensor> {
  PYBIND11_TYPE_CASTER(ms::Tensor, _("Tensor"));
  bool load(handle src, bool);
  static handle cast(const ms::Tensor &src, return_value_policy, handle);
};
}  // namespace detail
}  // namespace pybind11
#endif  // MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
