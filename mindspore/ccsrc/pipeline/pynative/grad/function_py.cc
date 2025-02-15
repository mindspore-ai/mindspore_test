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

#include "pipeline/pynative/grad/function_py.h"
#include <unordered_map>
#include <utility>
#include "utils/log_adapter.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/grad/grad_utils.h"
#include "pipeline/pynative/grad/grad.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
ValuePtr ConvertOutputTensorList(const py::object &obj) {
  py::tuple tuple = py::cast<py::tuple>(obj);
  ValuePtrList res;
  res.reserve(tuple.size());
  for (size_t i = 0; i < tuple.size(); i++) {
    auto tensor = parse::ConvertTensor(tuple[i]);
    if (tensor == nullptr) {
      res.emplace_back(kNone);
    } else {
      if (tensor->isa<tensor::BaseTensor>()) {
        tensor->cast<tensor::BaseTensorPtr>()->set_need_pipeline_sync(true);
      }
      res.emplace_back(tensor);
    }
  }
  return std::make_shared<ValueTuple>(res);
}
}  // namespace

const char CUSTOM_FORWARD_NAME[] = "forward";
const char CUSTOM_BACKWARD_NAME[] = "backward";

static BaseTensorPtrSet parse_mark_dirty(const FunctionPtr &fptr) {
  // versions of modified tensors should be increased.
  BaseTensorPtrSet dirty;
  py::object dirty_tensors = fptr->dirty_tensors();
  if (!dirty_tensors) {
    return dirty;
  }
  if (!py::isinstance<py::tuple>(dirty_tensors)) {
    MS_LOG(EXCEPTION) << "dirty_tensors of functionbase should be a tuple, but get a " << dirty_tensors.get_type();
  }
  py::tuple dirty_tensors_tp = py::cast<py::tuple>(dirty_tensors);
  size_t num_dirty = dirty_tensors_tp.size();
  for (size_t i = 0; i < num_dirty; i++) {
    py::object elem = dirty_tensors_tp[i];
    if (!tensor::IsTensorPy(elem) && !IsStubTensor(elem)) {
      MS_LOG(EXCEPTION) << "element of dirty_tensors should be a tensor or subtensor, but get a " << elem.get_type();
    }
    auto tensor = parse::ConvertTensor(elem);
    auto value = PyNativeAlgo::Common::StubNodeToValue(tensor);
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    dirty.insert(base_tensor);
    base_tensor->BumpVersion();
  }
  fptr->set_dirty_tensors(py::none());
  return dirty;
}

static BaseTensorPtrSet parse_non_differentiable(const FunctionPtr &fptr) {
  BaseTensorPtrSet non_diff;
  py::object non_diff_obj = fptr->non_differentiable();
  if (!non_diff_obj) {
    return non_diff;
  }
  if (!py::isinstance<py::tuple>(non_diff_obj)) {
    MS_LOG(EXCEPTION) << "non_differentiable of functionbase should be a tuple, but get a " << non_diff_obj.get_type();
  }
  py::tuple non_diff_tp = py::cast<py::tuple>(non_diff_obj);
  size_t num_non_diff = non_diff_tp.size();
  for (size_t i = 0; i < num_non_diff; i++) {
    py::object elem = non_diff_tp[i];
    if (!tensor::IsTensorPy(elem) && !IsStubTensor(elem)) {
      MS_LOG(EXCEPTION) << "element of non_differentiable should be a tensor or subtensor, but get a "
                        << elem.get_type();
    }
    auto tensor = parse::ConvertTensor(elem);
    auto value = PyNativeAlgo::Common::StubNodeToValue(tensor);
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    non_diff.insert(base_tensor);
  }
  fptr->set_non_differentiable(py::none());
  return non_diff;
}

static BaseTensorPtrSet parse_to_save(const FunctionPtr &fptr) {
  BaseTensorPtrSet to_save_tensors;
  py::object to_save_obj = fptr->to_save();
  if (!to_save_obj) {
    return to_save_tensors;
  }
  if (!py::isinstance<py::tuple>(to_save_obj)) {
    MS_LOG(EXCEPTION) << "to_save of functionbase should be a tuple, but get a " << to_save_obj.get_type();
  }
  py::tuple to_save_tp = py::cast<py::tuple>(to_save_obj);
  size_t num_to_save = to_save_tp.size();
  for (size_t i = 0; i < num_to_save; i++) {
    py::object elem = to_save_tp[i];
    if (!tensor::IsTensorPy(elem) && !IsStubTensor(elem) && !py::isinstance<py::none>(elem)) {
      MS_LOG(EXCEPTION) << "element of to_save should be a tensor or subtensor, but get a " << elem.get_type();
    }
    if (py::isinstance<py::none>(elem)) {
      continue;
    }
    auto tensor = parse::ConvertTensor(elem);
    auto value = PyNativeAlgo::Common::StubNodeToValue(tensor);
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    to_save_tensors.insert(base_tensor);
  }
  return to_save_tensors;
}

class ForwardGradGuard {
 public:
  explicit ForwardGradGuard(const GradExecutorPtr ptr) : ptr_(ptr), grad_flag_(ptr->grad_flag()) {
    ptr->set_grad_flag(false);
  }
  ~ForwardGradGuard() { ptr_->set_grad_flag(grad_flag_); }

 private:
  GradExecutorPtr ptr_;
  bool grad_flag_;
};

void UpdateTensorSetIfNeeded(const std::shared_ptr<FunctionContext> &context, tensor::BaseTensorPtr old_value,
                             tensor::BaseTensorPtr new_value) {
  if (context->input_base_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update input old: " << old_value << " new: " << new_value;
    context->input_base_tensors.erase(old_value);
    context->input_base_tensors.insert(new_value);
  }
  if (context->dirty_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update dirty old: " << old_value << " new: " << new_value;
    context->dirty_tensors.erase(old_value);
    context->dirty_tensors.insert(new_value);
  }
  if (context->non_diff_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update non_diff old: " << old_value << " new: " << new_value;
    context->non_diff_tensors.erase(old_value);
    context->non_diff_tensors.insert(new_value);
  }
}

void CleanBackwardUnusedTensorDeviceAddress(const std::shared_ptr<FunctionContext> &context) {
  std::unordered_map<tensor::BaseTensorPtr, tensor::BaseTensorPtr> changed;
  for (size_t i = 0; i < context->inputs.size(); i++) {
    if (context->inputs[i]->isa<tensor::BaseTensor>()) {
      auto base_tensor = context->inputs[i]->cast<tensor::BaseTensorPtr>();
      if (context->to_save_tensors.count(base_tensor) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::BaseTensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::BaseTensorPtr>());
        context->inputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean input tensor address, index: " << i;
      }
    }
  }
  for (size_t i = 0; i < context->flatten_outputs.size(); i++) {
    if (context->flatten_outputs[i]->isa<tensor::BaseTensor>()) {
      auto base_tensor = context->flatten_outputs[i]->cast<tensor::BaseTensorPtr>();
      if (context->to_save_tensors.count(base_tensor) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::BaseTensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::BaseTensorPtr>());
        context->flatten_outputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean output tensor address, index: " << i;
      }
    }
  }
}

void ConstructContextAfterForward(const std::shared_ptr<FunctionContext> &context, const FunctionPtr &ctx,
                                  const py::object &outputs) {
  // Convert output object to tensors.
  context->outputs = ConvertOutputTensorList(outputs);
  context->outputs = PyNativeAlgo::Common::StubNodeToValue(context->outputs);
  MS_LOG(DEBUG) << "function base info, has dirty_tensors: " << static_cast<bool>(ctx->dirty_tensors())
                << "has non_differentiable" << static_cast<bool>(ctx->non_differentiable());
  // Convert object use decided to tensors.
  context->dirty_tensors = parse_mark_dirty(ctx);
  context->non_diff_tensors = parse_non_differentiable(ctx);
  context->to_save_tensors = parse_to_save(ctx);
  MS_LOG(DEBUG) << "Parse info, dirty size: " << context->dirty_tensors.size()
                << ", non_diff size: " << context->non_diff_tensors.size()
                << "to_save size: " << context->to_save_tensors.size();

  // Convert input object to tensors.
  BaseTensorPtrSet input_base_tensors;
  input_base_tensors.reserve(context->inputs.size());
  for (size_t i = 0; i < context->inputs.size(); ++i) {
    if (!context->inputs[i]->isa<None>()) {
      (void)context->input_value_grad_type.emplace_back(
        PyNativeAlgo::AutoGradUtil::SetValueGradInfo(context->inputs[i], InputType::kConstant));
      auto value = PyNativeAlgo::Common::StubNodeToValue(context->inputs[i]);
      auto base_tensor = value->cast<tensor::BaseTensorPtr>();
      input_base_tensors.insert(base_tensor);
    }
  }
  context->input_base_tensors = input_base_tensors;
  (void)PyNativeAlgo::AutoGradUtil::SetValueGradInfo(context->outputs, InputType::kOpOutput);
  context->flatten_outputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(context->outputs);
}

py::object FunctionBase::apply(const py::object &cls, const py::args &inputs) {
  MS_LOG(DEBUG) << "enter apply function.";
  auto context = std::make_shared<FunctionContext>();
  py::function forward_fn = py::getattr(cls, CUSTOM_FORWARD_NAME);
  context->backward_fn = py::getattr(cls, CUSTOM_BACKWARD_NAME);
  // New a python object.
  context->obj = cls();
  context->inputs.reserve(inputs.size());

  auto ctx = py::cast<FunctionPtr>(context->obj);
  MS_EXCEPTION_IF_NULL(ctx);
  std::vector<bool> is_tensor_input;
  is_tensor_input.reserve(inputs.size());
  py::tuple need_grad_input = py::tuple(inputs.size());
  runtime::Pipeline::Get().WaitBpropStage();  // wait to get inputs value
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = parse::ConvertTensor(inputs[i]);
    if (tensor != nullptr) {
      tensor = PyNativeAlgo::Common::StubNodeToValue(tensor);
      (void)is_tensor_input.emplace_back(true);
      if (tensor->isa<tensor::BaseTensor>()) {
        tensor->cast<tensor::BaseTensorPtr>()->set_need_pipeline_sync(true);
      }
      auto base_tensor = tensor->cast<tensor::BaseTensorPtr>();
      need_grad_input[i] = PyNativeAlgo::AutoGradUtil::NeedGrad(base_tensor) ? py::bool_(true) : py::bool_(false);
      (void)context->inputs.emplace_back(tensor);
    } else {
      (void)is_tensor_input.emplace_back(false);
      need_grad_input[i] = py::bool_(false);
      (void)context->inputs.emplace_back(kNone);
    }
  }
  ctx->set_is_tensor_input(is_tensor_input);
  ctx->set_needs_input_grad(need_grad_input);

  // Call forward function.
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &grad_executor = pynative_executor->grad_executor();
  py::object outputs;
  {
    ForwardGradGuard guard(grad_executor);
    outputs = forward_fn(context->obj, *inputs);
  }
  bool modified = ensure_obj_tuple(&outputs);

  if (!grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "no need to do grad.";
    if (modified) {
      return py::cast<py::tuple>(outputs)[0];
    } else {
      return outputs;
    }
  }
  ConstructContextAfterForward(context, ctx, outputs);
  ValuePtrList flatten_outputs = context->flatten_outputs;
  BaseTensorPtrSet non_diff_tensors = context->non_diff_tensors;
  // Clean device address to reduce the occupation of resources.
  CleanBackwardUnusedTensorDeviceAddress(context);

  const auto &forward_executor = pynative_executor->forward_executor();
  runtime::Pipeline::Get().WaitFrontend();
  if (forward_executor->enable_async()) {
    auto auto_grad_cell_ptr = grad_executor->top_cell()->auto_grad_cell_ptr();
    auto task = [auto_grad_cell_ptr, new_context = std::move(context)]() mutable {
      (void)auto_grad_cell_ptr->CallCustomFunction(new_context);
    };
    grad_executor->DispatchGradQueueTask(std::move(task));
  } else {
    auto auto_grad_cell_ptr = grad_executor->top_cell()->auto_grad_cell_ptr();
    (void)auto_grad_cell_ptr->CallCustomFunction(context);
  }
  size_t num_output = (py::cast<py::tuple>(outputs)).size();
  py::tuple output_ret(num_output);
  MS_LOG(DEBUG) << "Output info, modified: " << modified << ", num_output: " << num_output;
  for (size_t i = 0; i < num_output; ++i) {
    if (flatten_outputs[i]->isa<tensor::BaseTensor>()) {
      auto base_tensor = flatten_outputs[i]->cast<tensor::BaseTensorPtr>();
      bool is_diff = non_diff_tensors.count(base_tensor) == 0;
      if (!is_diff) {
        // For tensor not need grad, we should clean grad meta data.
        base_tensor = std::make_shared<tensor::BaseTensor>(*base_tensor);
        base_tensor->set_auto_grad_meta_data(nullptr);
        output_ret[i] = CTensorToPyStubNodes(base_tensor);
      } else {
        output_ret[i] = CTensorToPyStubNodes(flatten_outputs[i]);
      }
    } else {
      output_ret[i] = (py::cast<py::tuple>(outputs))[i];
    }
  }
  MS_LOG(DEBUG) << "Leave apply function.";
  if (modified) {
    return output_ret[0];
  } else {
    return output_ret;
  }
}

void RegFunctionBase(const py::module *m) {
  (void)py::class_<FunctionBase, std::shared_ptr<FunctionBase>>(*m, "FunctionBase")
    .def(py::init<>())
    .def_static("apply", &FunctionBase::apply, "functionbase apply interface.")
    .def_property("needs_input_grad", &FunctionBase::needs_input_grad, &FunctionBase::set_needs_input_grad)
    .def_property("to_save", &FunctionBase::to_save, &FunctionBase::set_to_save)
    .def_property("non_differentiable", &FunctionBase::non_differentiable, &FunctionBase::set_non_differentiable)
    .def_property("dirty_tensors", &FunctionBase::dirty_tensors, &FunctionBase::set_dirty_tensors);
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
