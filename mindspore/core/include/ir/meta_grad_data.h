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
class Variable;
}  // namespace pynative::autograd

class TensorBackwardHook;
using TensorBackwardHookPtr = std::shared_ptr<TensorBackwardHook>;
using VariablePtr = std::shared_ptr<pynative::autograd::Variable>;
using VariableWeakPtr = std::weak_ptr<pynative::autograd::Variable>;

class AutoGradMetaInterface {
 public:
  [[nodiscard]] virtual VariablePtr UnsafeGetVariableImpl() const = 0;
  virtual void set_variable(const VariablePtr &variable) = 0;
  [[nodiscard]] virtual ParameterPtr parameter() const = 0;
  virtual void set_parameter(const ParameterPtr &parameter) = 0;
  virtual void set_k_node(const AnfNodePtr &k_node) = 0;
  [[nodiscard]] virtual AnfNodePtr k_node() const = 0;
  [[nodiscard]] virtual InputType input_type() const = 0;
  virtual void set_input_type(InputType input_type) = 0;
  [[nodiscard]] virtual size_t op_index() const = 0;
  virtual void set_op_index(size_t op_index) = 0;
  [[nodiscard]] virtual size_t output_index() const = 0;
  virtual void set_output_index(size_t output_index) = 0;
  virtual void Reset() = 0;
  virtual ~AutoGradMetaInterface() = default;
  virtual const std::map<uint64_t, TensorBackwardHookPtr> &backward_hooks() = 0;
  virtual bool is_register_hook() const = 0;
};
using AutoGradMetaInterfacePtr = std::shared_ptr<AutoGradMetaInterface>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_META_GRAD_DATA_H_
