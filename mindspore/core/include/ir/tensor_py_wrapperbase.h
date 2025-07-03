/**
 * Copyright 2025-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_TENSORPY_WRAPPERBASE_H_
#define MINDSPORE_CORE_IR_TENSORPY_WRAPPERBASE_H_

#include <memory>

#include "ir/tensor.h"

namespace mindspore {
namespace tensor {
// TensorPyWrapperBase: An abstract class
class MS_CORE_API TensorPyWrapperBase : public Value {
 public:
  TensorPyWrapperBase() = default;

  /// \brief Create TensorPyWrapperBase with Tensor.
  ///
  /// \param[in] input [TensorPtr] The given Tensor.
  explicit TensorPyWrapperBase(const TensorPtr input) : tensor_(input) {}

  /// Destructor of TensorPy.
  ~TensorPyWrapperBase() = default;

  MS_DECLARE_PARENT(TensorPyWrapperBase, Value);

  /// \brief Get the Tensor.
  ///
  /// \return The created Tensor.
  const TensorPtr &GetTensorWrapper() const { return tensor_; }

  /// \brief Set the Tensor.
  ///
  /// \param[in] base_tensor [TensorPtr] The given Tensor.
  void SetTensorWrapper(const TensorPtr &base_tensor) { tensor_ = base_tensor; }

 private:
  TensorPtr tensor_;
};

using TensorPyWrapperBasePtr = std::shared_ptr<TensorPyWrapperBase>;

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSORPY_WRAPPERBASE_H_
