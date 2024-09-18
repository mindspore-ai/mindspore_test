/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_

#include <string>
#include <utility>
#include <memory>
#include <vector>
#include "ir/anf.h"
#include "include/backend/kernel_graph.h"
#include "include/common/expander/core/node.h"
#include "pipeline/pynative/grad/variable.h"
#include "pybind11/pybind11.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace pynative {
namespace autograd {
struct CustomContext {
  // Custom cell name
  std::string cell_name;
  // Cell inputs
  ValuePtrList inputs;
  // Cell output
  ValuePtr output;
  // Input grad type
  std::vector<InputType> input_value_grad_type;
  // Custom bprop function
  py::function bprop_fn;
  // Python inputs for bprop_fn
  py::object original_inputs;
  // Python output
  py::object original_output;
  // Recompute weight size
  size_t weight_size{0};
  // Whether the cell is recompute cell
  bool is_recompute;
  ~CustomContext() {
    py::gil_scoped_acquire gil_acquire;
    bprop_fn = py::none();
    original_inputs = py::none();
    original_output = py::none();
  }
};

class CustomBackward : public BackwardNode {
 public:
  CustomBackward(string name, py::function bprop_fn, py::list bprop_inputs, abstract::AbstractBasePtr out_abstract,
                 bool is_recompute = false, size_t output_size = 1)
      : BackwardNode(std::move(name), output_size),
        bprop_fn_(std::move(bprop_fn)),
        bprop_inputs_(std::move(bprop_inputs)),
        out_abstract_(std::move(out_abstract)),
        is_recompute_(is_recompute) {}
  ~CustomBackward() override;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  void Release() override;

 private:
  py::function bprop_fn_;
  py::object bprop_inputs_;
  abstract::AbstractBasePtr out_abstract_;
  bool is_recompute_{false};
};
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_
