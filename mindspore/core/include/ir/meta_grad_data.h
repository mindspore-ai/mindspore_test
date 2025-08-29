/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_META_GRAD_DATA_H_
#define MINDSPORE_CORE_IR_META_GRAD_DATA_H_

#include <memory>
#include <utility>
#include <map>
#include <string>
#include "ir/anf.h"

namespace mindspore {

// For expander and pynative grad graph
enum class InputType {
  // Scala or Constant tensor, no need to grad
  kConstant = 0,
  // Weight parameter tensor
  kParameter,
  // Net input tensor
  kInput,
  // Other op output tensor
  kOpOutput,
  // Default
  kUnkown,
};

namespace pynative::autograd {
class BackwardNode;
}  // namespace pynative::autograd
using BackwardNodePtr = std::shared_ptr<pynative::autograd::BackwardNode>;

namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<tensor::Tensor>;
}  // namespace tensor

class GradHookInterface {
 public:
  [[nodiscard]] virtual bool requires_grad(const tensor::TensorPtr &self) const = 0;
  virtual void set_requires_grad(const tensor::TensorPtr &self, bool requires_grad) = 0;
  [[nodiscard]] virtual bool retains_grad(const tensor::TensorPtr &self) const = 0;
  virtual void retain_grad(const tensor::TensorPtr &self) = 0;
  [[nodiscard]] virtual tensor::TensorPtr grad(const tensor::TensorPtr &self) const = 0;
  virtual void set_grad(const tensor::TensorPtr &self, const tensor::TensorPtr &grad) = 0;
  [[nodiscard]] virtual BackwardNodePtr grad_node(const tensor::TensorPtr &self) const = 0;
  [[nodiscard]] virtual bool is_leaf(const tensor::TensorPtr &self) const = 0;
  [[nodiscard]] virtual size_t output_index(const tensor::TensorPtr &self) const = 0;
  virtual ~GradHookInterface() = default;
};
using GradHookInterfacePtr = std::unique_ptr<GradHookInterface>;

class AutoGradMetaInterface {
 public:
  [[nodiscard]] virtual BackwardNodePtr UnsafeGetGradNodeImpl() const = 0;
  virtual void set_grad_node(const BackwardNodePtr &variable) = 0;
  [[nodiscard]] virtual InputType input_type() const = 0;
  virtual void set_input_type(InputType input_type) = 0;
  [[nodiscard]] virtual size_t output_index() const = 0;
  virtual void set_output_index(size_t output_index) = 0;
  [[nodiscard]] virtual bool requires_grad() const = 0;
  virtual void set_requires_grad(bool requires_grad) = 0;
  [[nodiscard]] virtual const tensor::TensorPtr &grad() const = 0;
  virtual void set_grad(const tensor::TensorPtr &update_grad) = 0;
  [[nodiscard]] virtual bool retains_grad() const = 0;
  virtual void set_retains_grad(bool retains_grad) = 0;
  virtual bool is_view() const = 0;
  virtual ~AutoGradMetaInterface() = default;
};
using AutoGradMetaInterfacePtr = std::shared_ptr<AutoGradMetaInterface>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_META_GRAD_DATA_H_
