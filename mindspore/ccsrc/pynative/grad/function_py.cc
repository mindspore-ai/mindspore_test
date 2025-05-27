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

#include "pynative/grad/function_py.h"
#include <string>
#include <unordered_map>
#include <utility>
#include "utils/log_adapter.h"
#include "pynative/pynative_utils.h"
#include "pynative/grad/grad_utils.h"
#include "pynative/grad/function/func_grad.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/pynative/grad_state.h"
#include "include/common/pynative/common_utils.h"
#include "pyboost/functions/auto_grad_guard.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
ValuePtrList ConvertOutputTensorList(const py::object &obj) {
  py::tuple tuple = py::cast<py::tuple>(obj);
  ValuePtrList res;
  res.reserve(tuple.size());
  for (size_t i = 0; i < tuple.size(); i++) {
    auto tensor = tensor::ConvertToTensor(tuple[i]);
    if (tensor == nullptr) {
      res.emplace_back(kNone);
    } else {
      tensor->set_need_pipeline_sync(true);
      res.emplace_back(tensor);
    }
  }
  return res;
}

TensorPtr view_as_self_with_no_grad(const TensorPtr &self) {
  kernel::pyboost::OpStatus status{false, false, 0,
                                   MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET)};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  return kernel::pyboost::view(self, self->shape());
}
}  // namespace

const char CUSTOM_FORWARD_NAME[] = "forward";
const char CUSTOM_BACKWARD_NAME[] = "backward";

static TensorPtrSet parse_mark_dirty(const FunctionPtr &fptr) {
  // versions of modified tensors should be increased.
  TensorPtrSet dirty;
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
    if (!tensor::IsTensorPy(elem)) {
      MS_LOG(EXCEPTION) << "element of dirty_tensors should be a tensor, but get a " << elem.get_type();
    }
    auto base_tensor = tensor::ConvertToTensor(elem);
    MS_EXCEPTION_IF_NULL(base_tensor);
    dirty.insert(base_tensor);
    base_tensor->BumpVersion();
  }
  fptr->set_dirty_tensors(py::none());
  return dirty;
}

static TensorPtrSet parse_non_differentiable(const FunctionPtr &fptr) {
  TensorPtrSet non_diff;
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
    if (!tensor::IsTensorPy(elem)) {
      MS_LOG(EXCEPTION) << "element of non_differentiable should be a tensor, but get a " << elem.get_type();
    }
    auto base_tensor = tensor::ConvertToTensor(elem);
    MS_EXCEPTION_IF_NULL(base_tensor);
    non_diff.insert(base_tensor);
  }
  fptr->set_non_differentiable(py::none());
  return non_diff;
}

static TensorPtrSet parse_to_save(const FunctionPtr &fptr) {
  TensorPtrSet to_save_tensors;
  py::object to_save_obj = fptr->saved_tensors();
  if (!to_save_obj) {
    return to_save_tensors;
  }
  if (!py::isinstance<py::tuple>(to_save_obj)) {
    MS_LOG(EXCEPTION) << "saved_tensors of functionbase should be a tuple, but get a " << to_save_obj.get_type();
  }
  py::tuple to_save_tp = py::cast<py::tuple>(to_save_obj);
  size_t num_to_save = to_save_tp.size();
  for (size_t i = 0; i < num_to_save; i++) {
    py::object elem = to_save_tp[i];
    if (!tensor::IsTensorPy(elem) && !py::isinstance<py::none>(elem)) {
      MS_LOG(EXCEPTION) << "element of to_save should be a tensor, but get a " << elem.get_type();
    }
    if (py::isinstance<py::none>(elem)) {
      continue;
    }
    auto base_tensor = tensor::ConvertToTensor(elem);
    to_save_tensors.insert(base_tensor);
  }
  return to_save_tensors;
}

class ForwardGradGuard {
 public:
  explicit ForwardGradGuard(const GradExecutorPtr ptr) : ptr_(ptr), grad_flag_(GradState::Get().grad_flag()) {
    GradState::Get().set_grad_flag(false);
  }
  ~ForwardGradGuard() { GradState::Get().set_grad_flag(grad_flag_); }

 private:
  GradExecutorPtr ptr_;
  bool grad_flag_;
};

void UpdateTensorSetIfNeeded(const std::shared_ptr<FunctionContext> &context, tensor::TensorPtr old_value,
                             tensor::TensorPtr new_value) {
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
  std::unordered_map<tensor::TensorPtr, tensor::TensorPtr> changed;
  for (size_t i = 0; i < context->inputs.size(); i++) {
    if (context->inputs[i]->isa<tensor::Tensor>()) {
      auto base_tensor = context->inputs[i]->cast<tensor::TensorPtr>();
      if (context->to_save_tensors.count(base_tensor) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::TensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::TensorPtr>());
        context->inputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean input tensor address, index: " << i;
      }
    }
  }
  for (size_t i = 0; i < context->flatten_outputs.size(); i++) {
    if (context->flatten_outputs[i]->isa<tensor::Tensor>()) {
      auto base_tensor = context->flatten_outputs[i]->cast<tensor::TensorPtr>();
      if (context->to_save_tensors.count(base_tensor) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::TensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::TensorPtr>());
        context->flatten_outputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean output tensor address, index: " << i;
      }
    }
  }
}

void ConstructContextAfterForward(const std::shared_ptr<FunctionContext> &context, const FunctionPtr &ctx,
                                  const py::object &outputs) {
  // Convert output object to tensors.
  context->flatten_outputs = ConvertOutputTensorList(outputs);
  MS_LOG(DEBUG) << "function base info, has dirty_tensors: " << static_cast<bool>(ctx->dirty_tensors())
                << "has non_differentiable" << static_cast<bool>(ctx->non_differentiable());
  // Convert object use decided to tensors.
  context->dirty_tensors = parse_mark_dirty(ctx);
  context->non_diff_tensors = parse_non_differentiable(ctx);
  context->to_save_tensors = parse_to_save(ctx);
  MS_LOG(DEBUG) << "Parse info, dirty size: " << context->dirty_tensors.size()
                << ", non_diff size: " << context->non_diff_tensors.size()
                << "saved_tensors size: " << context->to_save_tensors.size();

  // Convert input object to tensors.
  TensorPtrSet input_base_tensors;
  input_base_tensors.reserve(context->inputs.size());
  for (size_t i = 0; i < context->inputs.size(); ++i) {
    const auto &input_value = context->inputs[i];
    if (!input_value->isa<None>()) {
      (void)context->input_value_grad_type.emplace_back(
        AutoGradUtil::SetValueGradInfo(input_value, InputType::kConstant));
      auto base_tensor = input_value->cast<tensor::TensorPtr>();
      input_base_tensors.insert(base_tensor);
    }
  }
  context->input_base_tensors = input_base_tensors;
}

py::object FunctionBase::saved_tensors() const {
  if (!saved_tensors_) {
    return py::tuple();
  }
  if (saved_nodes_.empty()) {
    return py::cast<py::tuple>(saved_tensors_);
  }
  auto tensors = py::cast<py::list>(saved_tensors_);
  py::tuple saved_tensors(tensors.size());
  auto grad_node = weak_grad_node_.lock();
  MS_EXCEPTION_IF_NULL(grad_node);
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto iter = saved_nodes_.find(i);
    if (iter == saved_nodes_.end()) {
      saved_tensors[i] = tensors[i];
      continue;
    }
    const auto &saved_node = iter->second;
    const auto tensor = saved_node->Unwrap(grad_node)->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      saved_tensors[i] = py::none();
    } else {
      saved_tensors[i] = py::reinterpret_steal<py::object>(tensor::PackTensor(tensor));
    }
  }
  return std::move(saved_tensors);
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
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();  // wait to get inputs value
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto tensor = tensor::ConvertToTensor(inputs[i]);
    if (tensor != nullptr) {
      (void)is_tensor_input.emplace_back(true);
      tensor->set_need_pipeline_sync(true);
      need_grad_input[i] = AutoGradUtil::NeedGrad(tensor) ? py::bool_(true) : py::bool_(false);
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

  if (!GradState::Get().RequiresGrad()) {
    MS_LOG(DEBUG) << "no need to do grad.";
    if (modified) {
      return py::cast<py::tuple>(outputs)[0];
    } else {
      return outputs;
    }
  }

  runtime::Pipeline::Get().WaitFrontend();

  ConstructContextAfterForward(context, ctx, outputs);
  auto &flatten_outputs = context->flatten_outputs;
  const auto &non_diff_tensors = context->non_diff_tensors;
  const auto &input_tensor_set = context->input_base_tensors;
  const auto &dirty_tensor_set = context->dirty_tensors;

  size_t num_output = (py::cast<py::tuple>(outputs)).size();
  py::tuple output_ret(num_output);
  MS_LOG(DEBUG) << "Output info, modified: " << modified << ", num_output: " << num_output;
  for (size_t i = 0; i < num_output; ++i) {
    if (flatten_outputs[i]->isa<tensor::Tensor>()) {
      auto tensor = flatten_outputs[i]->cast<tensor::TensorPtr>();
      bool is_diff = non_diff_tensors.count(tensor) == 0;
      if (!is_diff) {
        // For tensor not need grad, we should clean grad meta data.
        tensor = std::make_shared<tensor::Tensor>(*tensor);
        tensor->set_auto_grad_meta_data(nullptr);
        output_ret[i] = CValueToPybindObj(tensor);
      } else {
        if (input_tensor_set.count(tensor) > 0) {
          if (dirty_tensor_set.count(tensor) == 0) {
            tensor = view_as_self_with_no_grad(tensor);
            flatten_outputs[i] = tensor;
          }
        }
        AutoGradUtil::SetValueGradInfo(tensor, InputType::kOpOutput);
        output_ret[i] = CValueToPybindObj(tensor);
      }
    } else {
      output_ret[i] = (py::cast<py::tuple>(outputs))[i];
    }
  }

  // Clean device address to reduce the occupation of resources.
  CleanBackwardUnusedTensorDeviceAddress(context);
  // Generate saved nodes， and clear saved tensor.
  ctx->GenerateSavedNodes(context);

  const auto &forward_executor = pynative_executor->forward_executor();
  if (forward_executor->enable_async()) {
    auto task = [new_context = std::move(context)]() mutable { (void)CallCustomPyFunction(new_context); };
    grad_executor->DispatchGradQueueTask(std::move(task));
  } else {
    (void)CallCustomPyFunction(context);
  }

  MS_LOG(DEBUG) << "Leave apply function.";
  if (modified) {
    return output_ret[0];
  }
  return output_ret;
}

void FunctionBase::GenerateSavedNodes(const std::shared_ptr<FunctionContext> &ctx) {
  if (!saved_tensors_) {
    return;
  }
  py::list tensors = py::cast<py::list>(saved_tensors_);
  if (!tensors) {
    MS_LOG(EXCEPTION) << "save tensor should be tuple!";
  }
  auto check_is_output = [&ctx](const ValuePtr &val) {
    return std::any_of(ctx->flatten_outputs.begin(), ctx->flatten_outputs.end(),
                       [&val](const ValuePtr &output) { return val.get() == output.get(); });
  };
  saved_nodes_.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &obj = tensors[i];
    if (tensor::IsTensorPy(obj)) {
      auto tensor = tensor::ConvertToTensor(obj);
      // Now custom function not support high order, this need to do later.
      if (check_is_output(tensor)) {
        saved_nodes_[i] = std::make_shared<SavedNode>(CommonUtils::ShallowCopyAndDetach(tensor), nullptr, false, true);
        tensors[i] = py::object();
      }
    } else if (!py::isinstance<py::none>(obj)) {
      MS_LOG(EXCEPTION)
        << "Please check your custom function, that save_for_backward() only support None and tensor, but got "
        << py::str(obj);
    }
  }
}

void RegFunctionBase(const py::module *m) {
  (void)py::class_<FunctionBase, std::shared_ptr<FunctionBase>>(*m, "FunctionBase")
    .def(py::init<>())
    .def_static("apply", &FunctionBase::apply, "functionbase apply interface.")
    .def_property("needs_input_grad", &FunctionBase::needs_input_grad, &FunctionBase::set_needs_input_grad)
    .def_property("saved_tensors", &FunctionBase::saved_tensors, &FunctionBase::set_saved_tensors)
    .def_property("non_differentiable", &FunctionBase::non_differentiable, &FunctionBase::set_non_differentiable)
    .def_property("dirty_tensors", &FunctionBase::dirty_tensors, &FunctionBase::set_dirty_tensors)
    .def_property("materialize_grads", &FunctionBase::materialize_grads, &FunctionBase::set_materialize_grads);
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
