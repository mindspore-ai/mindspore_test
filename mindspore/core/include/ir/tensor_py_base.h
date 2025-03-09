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

#ifndef MINDSPORE_CORE_IR_TENSORPY_BASE_H_
#define MINDSPORE_CORE_IR_TENSORPY_BASE_H_

#include <memory>

#include "ir/base_tensor.h"

namespace mindspore {
namespace tensor {
// TensorPyBase: An abstract class
class MS_CORE_API TensorPyBase : public Value {
 public:
  TensorPyBase() = default;

  /// \brief Create TensorPyBase with BaseTensor.
  ///
  /// \param[in] input [BaseTensorPtr] The given BaseTensor.
  explicit TensorPyBase(const BaseTensorPtr &input) : tensor_(input) {}

  /// Destructor of TensorPy.
  ~TensorPyBase() override = default;

  MS_DECLARE_PARENT(TensorPyBase, Value);

  /// \brief Get the BaseTensor.
  ///
  /// \return The created BaseTensor.
  virtual BaseTensorPtr GetBaseTensor() const = 0;

  /// \brief Set the BaseTensor.
  ///
  /// \param[in] base_tensor [BaseTensorPtr] The given BaseTensor.
  void SetBaseTensor(const BaseTensorPtr &base_tensor) { tensor_ = base_tensor; }

 protected:
  BaseTensorPtr tensor_{nullptr};
};

using TensorPyBasePtr = std::shared_ptr<TensorPyBase>;

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSORPY_BASE_H_
