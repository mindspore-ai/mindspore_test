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
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "include/common/pynative/variable.h"
#include "mindspore/ccsrc/pynative/backward/grad_utils.h"

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
  // Recompute weight size
  size_t weight_size{0};
  // Whether the cell is recompute cell
  bool is_recompute;
  ~CustomContext() {
    py::gil_scoped_acquire gil_acquire;
    bprop_fn = py::object();
    original_inputs = py::object();
  }
};

class CustomBackward : public BackwardNode {
 public:
  CustomBackward(string name, py::function bprop_fn, py::list bprop_inputs, SavedNodePtr saved_output,
                 const std::vector<TensorMeta> &input_meta, abstract::AbstractBasePtr out_abstract,
                 bool is_recompute = false, size_t output_size = 1)
      : BackwardNode(std::move(name), output_size),
        bprop_fn_(std::move(bprop_fn)),
        bprop_inputs_(std::move(bprop_inputs)),
        saved_output_(std::move(saved_output)),
        input_meta_(input_meta),
        out_abstract_(std::move(out_abstract)),
        is_recompute_(is_recompute) {}
  ~CustomBackward() override;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;
  void Release() override;

 private:
  py::function bprop_fn_;
  py::object bprop_inputs_;
  SavedNodePtr saved_output_;
  std::vector<TensorMeta> input_meta_;
  abstract::AbstractBasePtr out_abstract_;
  bool is_recompute_{false};
};

class PyBackwardNode : public BackwardNode {
 public:
  PyBackwardNode(string name, py::function backward_fn, py::object obj, std::vector<TensorMeta> input_meta,
                 size_t output_size = 1)
      : BackwardNode(std::move(name), output_size),
        backward_fn_(std::move(backward_fn)),
        obj_(std::move(obj)),
        input_meta_(std::move(input_meta)) {}
  ~PyBackwardNode() override;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;
  void Release() override;
  void SetOutAbstract(abstract::AbstractBasePtr out_abstract) { out_abstract_ = std::move(out_abstract); }
  void SetOutputSize(size_t output_size) { output_size_ = output_size; }

 private:
  py::function backward_fn_;
  py::object obj_;
  std::vector<TensorMeta> input_meta_;
  abstract::AbstractBasePtr out_abstract_;
};
using PyBackwardNodePtr = std::shared_ptr<PyBackwardNode>;

}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_
