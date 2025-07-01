/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"
#include "pybind11/detail/init.h"
#include "pybind_api/pybind_patch.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/ccsrc/include/common/utils/tensor_py.h"
#include "mindspore/ccsrc/pynative/grad/function/func_grad.h"

namespace mindspore {
namespace pynative {
namespace autograd {
std::vector<ValuePtr> ConvertPyTupleToTensorList(const py::tuple &tuple_args) {
  std::vector<ValuePtr> tensors;
  tensors.reserve(tuple_args.size());
  for (size_t i = 0; i < tuple_args.size(); ++i) {
    py::object obj = tuple_args[i];
    if (!tensor::IsTensorPy(obj)) {
      MS_LOG(EXCEPTION) << "Elements of tuple should be tensor, but got" << py::str(obj);
    }
    auto tensor = tensor::ConvertToTensor(obj);
    (void)tensors.emplace_back(tensor);
  }
  return tensors;
}

py::object RunBackward(const py::object &tensors, const py::object &grad_tensors, bool keep_graph, bool high_order,
                       const py::object &inputs, bool allow_unreachable, bool accumulate_grad) {
  runtime::Pipeline::Get().WaitAll();
  if (!py::isinstance<py::tuple>(tensors)) {
    MS_LOG(EXCEPTION) << "Output tensors should be tuple! but got " << py::str(tensors);
  }
  ValuePtr output = std::make_shared<ValueTuple>(ConvertPyTupleToTensorList(tensors));
  ValuePtr sens_gradients = nullptr;
  if (py::isinstance<py::tuple>(grad_tensors)) {
    sens_gradients = std::make_shared<ValueTuple>(ConvertPyTupleToTensorList(grad_tensors));
  }
  if (!py::isinstance<py::tuple>(inputs) && !py::isinstance<py::none>(inputs)) {
    MS_LOG(EXCEPTION) << "input tensors should be tuple or none! but got " << py::str(inputs);
  }
  ValuePtrList input_tensors;
  if (py::isinstance<py::tuple>(inputs)) {
    input_tensors = ConvertPyTupleToTensorList(inputs);
    for (const auto &input_tensor : input_tensors) {
      auto tensor = input_tensor->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->retain_grad();
    }
  }
  auto engine = std::make_shared<autograd::AutoDiff>(output, keep_graph, high_order, false);
  autograd::AutoDiffGuard auto_diff_guard(engine);
  auto grads = engine->RunBackward(input_tensors, sens_gradients, accumulate_grad);
  engine->Clear();
  if (grads->isa<None>()) {
    return py::none();
  } else {
    auto tuple_grads = grads->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(tuple_grads);
    py::tuple py_grads(tuple_grads->size());
    for (size_t i = 0; i < tuple_grads->size(); ++i) {
      auto tensor = tuple_grads->value()[i]->cast<tensor::TensorPtr>();
      if (tensor == nullptr) {
        MS_LOG(EXCEPTION) << "Grads of tensors should be a tensor, but got " << tuple_grads->value()[i]->ToString();
      }
      py_grads[i] = tensor::PackTensor(tensor);
    }
    return std::move(py_grads);
  }
}

void RegBackwardFunction(py::module *m) {
  (void)m->def("run_backward", &RunBackward, py::arg("tensors"), py::arg("grad_tensors"), py::arg("keep_graph"),
               py::arg("create_graph"), py::arg("inputs"), py::kw_only(), py::arg("allow_unreachable") = True,
               py::arg("accumulate_grad") = True, "run backward function");
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
