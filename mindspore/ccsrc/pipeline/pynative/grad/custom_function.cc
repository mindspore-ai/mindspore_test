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

#include "pipeline/pynative/grad/custom_function.h"
#include "runtime/pipeline/pipeline.h"
#include "pipeline/pynative/grad/function/func_builder.h"
#include "pipeline/pynative/pynative_execute.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace pynative {
namespace autograd {
CustomBackward::~CustomBackward() {
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::none();
  bprop_inputs_ = py::none();
}

ValuePtrList CustomBackward::CallBackward(const ValuePtrList &grads) {
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(DEBUG) << "Begin CustomBackwardNode CallBackward ";
  auto gradient = PyNativeAlgo::DataConvert::ValueListToValue(grads, out_abstract_);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Python grad func can not process None, we need to convert None to zero tensor.
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  auto filled_zeros_grad = func_builder.FillZeros(gradient, out_abstract_);

  // Run bprop function.
  py::gil_scoped_acquire gil_acquire;
  auto py_grad = ValueToPyData(filled_zeros_grad);
  auto py_tensor_grad = ConvertCTensorToPyTensor(py_grad);
  py::list list_inputs = bprop_inputs_.cast<py::list>();
  list_inputs.append(py_tensor_grad);
  size_t non_inp_args_size = is_recompute_ ? kSizeOne : kSizeTwo;
  auto inp_args_size = list_inputs.size() - non_inp_args_size;
  py::tuple input_args(inp_args_size);
  for (size_t i = 0; i < inp_args_size; ++i) {
    input_args[i] = list_inputs[i];
  }
  py::tuple fn_args(list_inputs.size());
  for (size_t i = 0; i < fn_args.size(); ++i) {
    fn_args[i] = list_inputs[i];
  }
  const auto &inst = pynative::PyNativeExecutor::GetInstance();
  if (inst->grad_flag()) {
    inst->NewGraph(bprop_fn_, input_args.cast<py::args>());
  }
  py::object grads_obj = bprop_fn_(*fn_args);
  py::tuple input_grads = CheckBpropOut(grads_obj, fn_args, name());
  py::object out = grads_obj;
  // If grads.size() > inp_args_size, that means exist weights.
  if (input_grads.size() > inp_args_size) {
    MS_LOG(DEBUG) << "Get grads size " << input_grads.size();
    out = py::cast<py::tuple>(grads_obj)[0];
  }
  if (inst->grad_flag()) {
    inst->EndGraph(bprop_fn_, out, input_args.cast<py::args>());
  }
  MS_LOG(DEBUG) << "Run cell custom bprop function end.";
  ValuePtrList gradient_values;
  ConvertPyObjectToCTensor(input_grads, &gradient_values);
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn grad output is empty!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  runtime::Pipeline::Get().WaitForward();
  return gradient_tensors;
}

void CustomBackward::Release() {
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::none();
  bprop_inputs_ = py::none();
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
