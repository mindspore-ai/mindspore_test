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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/common/pynative/variable.h"
#include "include/common/visible.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/hook/custom_function.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;

inline bool ensure_obj_tuple(py::object *obj) {
  if (py::isinstance<py::tuple>(*obj)) {
    return false;
  }
  py::tuple tuple = py::make_tuple(*obj);
  if (!tuple) {
    MS_LOG(EXCEPTION) << "tuple is null.";
  }
  *obj = tuple;
  return true;
}

using TensorPtrSet = std::unordered_set<tensor::TensorPtr>;

struct FunctionContext {
  // The input of apply function
  ValuePtrList inputs;

  // The output of forward function in flatten format
  ValuePtrList flatten_outputs;

  // The input type of apply function input
  std::vector<InputType> input_value_grad_type;

  // Set of input tensors
  TensorPtrSet input_base_tensors;
  // Set of dirty tensors
  TensorPtrSet dirty_tensors;
  // Set of non_diff tensors
  TensorPtrSet non_diff_tensors;
  // Set of to_save tensors
  TensorPtrSet to_save_tensors;
  PyBackwardNodePtr grad_node;
};

class PYNATIVE_EXPORT FunctionBase {
 public:
  // The enter of custom function.
  static py::object apply(const py::object &cls, const py::args &inputs);

  py::object needs_input_grad() const { return needs_input_grad_; }

  void set_needs_input_grad(const py::object &needs_input_grad) { needs_input_grad_ = needs_input_grad; }

  py::object saved_tensors() const;

  void set_saved_tensors(const py::object &saved_tensors) { saved_tensors_ = py::cast<py::list>(saved_tensors); }

  py::object non_differentiable() { return non_differentiable_; }

  void set_non_differentiable(const py::object &non_differentiable) { non_differentiable_ = non_differentiable; }

  py::object dirty_tensors() { return dirty_tensors_; }

  void set_dirty_tensors(const py::object &dirty_tensors) { dirty_tensors_ = dirty_tensors; }

  bool materialize_grads() { return materialize_grads_; }

  void set_materialize_grads(const py::object &materialize_grads) {
    if (!py::isinstance<py::bool_>(materialize_grads)) {
      MS_LOG(EXCEPTION) << "set_materialize_grads need bool value, but get a " << materialize_grads.get_type();
    }
    materialize_grads_ = py::cast<bool>(materialize_grads);
  }

  std::vector<bool> is_tensor_input() { return is_tensor_input_; }

  void set_weak_grad_node(const BackwardNodePtr &grad_node) { weak_grad_node_ = grad_node; }

  void set_is_tensor_input(const std::vector<bool> &is_tensor_input) { is_tensor_input_ = is_tensor_input; }
  void GenerateSavedNodes(const std::shared_ptr<FunctionContext> &ctx);

 private:
  // A python tuple return to use to indicate whether inputs need grad.
  py::object needs_input_grad_;

  // The context carry tensors from forward function to backward function. Result of `save_for_backward` function.
  py::object saved_tensors_;

  // The tensors that are not differentiable decided by use. Result of `mark_non_differentiable` function.
  py::object non_differentiable_;

  // The tensor that have been modified. Result of `mark_dirty` function.
  py::object dirty_tensors_;

  // The flag indicate whether to materialize none output grad tensors into
  // tensors full of zeros.
  bool materialize_grads_ = true;

  // True is the input is tensor
  std::vector<bool> is_tensor_input_;

  // This is used for avoid cycle reference.
  std::unordered_map<size_t, SavedNodePtr> saved_nodes_;

  // This is used for unpack saved tensors.
  std::weak_ptr<BackwardNode> weak_grad_node_;
};

using FunctionPtr = std::shared_ptr<FunctionBase>;

PYNATIVE_EXPORT void RegFunctionBase(const py::module *m);

}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_
