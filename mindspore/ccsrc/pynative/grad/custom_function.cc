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

#include "pynative/grad/custom_function.h"
#include "pynative/grad/function_py.h"
#include "runtime/pipeline/pipeline.h"
#include "pynative/grad/function/func_builder.h"
#include "pynative/pynative_execute.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/tensor_py.h"
#include "pynative/pynative_utils.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
// The arguments of backward function are ctx and gradients correspongding to outputs of forward function.
py::tuple ConstructBackwardArgs(const py::object &ctx, const py::object &py_tensor_grad) {
  auto num_args = py::isinstance<py::tuple>(py_tensor_grad) ? 1 + py::cast<py::tuple>(py_tensor_grad).size() : 2;
  py::tuple res(num_args);
  res[0] = ctx;
  if (py::isinstance<py::tuple>(py_tensor_grad)) {
    py::tuple grad_tuple = py::cast<py::tuple>(py_tensor_grad);
    for (size_t i = 0; i < grad_tuple.size(); i++) {
      res[i + 1] = grad_tuple[i];
    }
  } else {
    res[1] = py_tensor_grad;
  }
  return res;
}

ValuePtr ValueListToValue(const ValuePtrList &list) {
  if (list.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "Value ptr list should not be empty";
  }
  if (list.size() == kSizeOne) {
    return list[kIndex0];
  }
  return std::make_shared<ValueTuple>(list);
}
}  // namespace

CustomBackward::~CustomBackward() {
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::object();
  bprop_inputs_ = py::object();
}

ValuePtrList CustomBackward::CallBackward(const ValuePtrList &grads) {
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin CustomBackwardNode CallBackward ";
  auto gradient = PyNativeAlgo::DataConvert::ValueListToValue(grads, out_abstract_);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Python grad func can not process None, we need to convert None to zero tensor.
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  auto filled_zeros_grad = func_builder.FillZeros(gradient, out_abstract_);

  // Run bprop function.
  py::gil_scoped_acquire gil_acquire;
  py::object py_tensor_grad = CValueToPybindObj(filled_zeros_grad);
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
  ConvertPybindTupleGradToCValue(input_grads, &gradient_values, true);
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn grad output is empty!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  runtime::Pipeline::Get().WaitFrontend();
  return gradient_tensors;
}

void CustomBackward::Release() {
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::object();
  bprop_inputs_ = py::object();
}

ValuePtrList PyBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin PyBackwardNode CallBackward";
  // Construct input for backward function.
  py::gil_scoped_acquire gil_acquire;
  auto gradients = ValueListToValue(grads);
  auto ctx = py::cast<FunctionPtr>(obj_);
  MS_EXCEPTION_IF_NULL(ctx);
  py::object py_tensor_grad;
  if (ctx->materialize_grads()) {
    const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    // Python grad func can not process None, we need to convert None to zero tensor.
    auto func_builder = FuncBuilder(name_, device_target, nullptr);
    auto filled_zeros_grad = func_builder.FillZeros(gradients, out_abstract_);
    py_tensor_grad = CValueToPybindObj(filled_zeros_grad);
  } else {
    py_tensor_grad = CValueToPybindObj(gradients);
  }
  MS_LOG(DEBUG) << "Args info, grad is tuple " << py::isinstance<py::tuple>(py_tensor_grad) << ", is tensor input size "
                << ctx->is_tensor_input().size() << "materialize_grads " << ctx->materialize_grads();

  py::tuple fn_args = ConstructBackwardArgs(obj_, py_tensor_grad);
  // Call python backward function.
  py::object grads_obj = backward_fn_(*fn_args);

  (void)ensure_obj_tuple(&grads_obj);
  auto grad_tuple = py::cast<py::tuple>(grads_obj);
  size_t num_backward_out = grad_tuple.size();
  size_t num_forward_in = ctx->is_tensor_input().size();
  if (num_backward_out != num_forward_in) {
    MS_LOG(EXCEPTION) << "Function backward return a wrong number of gradients, expect: " << num_forward_in
                      << "but: " << num_backward_out;
  }

  for (size_t i = 0; i < num_backward_out; i++) {
    bool is_tensor = (ctx->is_tensor_input())[i];
    py::object output = grad_tuple[i];
    // The gradient of Input that is not tensor should be none.
    if (!is_tensor && !py::isinstance<py::none>(output)) {
      MS_LOG(EXCEPTION) << "Input is not tensor, but gradient is not none, position: " << i
                        << " type: " << output.get_type();
    }
    // The gradient should be either none or tensor.
    if (!py::isinstance<py::none>(output) && !tensor::IsTensorPy(output)) {
      MS_LOG(EXCEPTION) << "Gradient should be none or tensor, position: " << i << " type: " << output.get_type();
    }
  }

  // Convert python object to tensor.
  ValuePtrList gradient_values;
  ConvertPybindTupleGradToCValue(grad_tuple, &gradient_values, true);
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Custom backward function output is empty!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "End PyBackwardNode CallBackward";
  return gradient_tensors;
}

void PyBackwardNode::Release() {
  py::gil_scoped_acquire gil_acquire;
  backward_fn_ = py::object();
  obj_ = py::object();
}

}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
