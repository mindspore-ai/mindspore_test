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

#include "pynative/grad/hook_py.h"
#include <memory>
#include <string>
#include "include/common/utils/hook.h"
#include "include/common/pynative/adapter.h"
#include "pipeline/jit/ps/pipeline.h"
#include "runtime/pipeline/pipeline.h"
#include "pynative/grad/grad_utils.h"
#include "pynative/grad/function/func_grad.h"

namespace mindspore::pynative::autograd {
namespace {
BackwardNodePtr BuildAutoGradMeta(const tensor::TensorPtr &tensor) {
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();
  auto auto_grad_meta_data = impl::GetAutogradMetaImpl(tensor);
  if (auto_grad_meta_data == nullptr) {
    if (tensor->param_info() != nullptr && !tensor->param_info()->requires_grad()) {
      MS_LOG(EXCEPTION) << "The tensor requires grad is false, which can not register tensor hook";
    }
    MS_LOG(DEBUG) << "Create leaf node for: " << tensor->ToString();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    auto fn = std::make_shared<autograd::LeafNode>(
      tensor->param_info() != nullptr ? tensor->param_info()->name() : "register_hook_input", tensor->shape(),
      tensor->Dtype(), tensor->is_parameter());
    auto_grad_meta_data->set_grad_node(fn);
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    return fn;
  }
  auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl();
  if (grad_node == nullptr) {
    MS_LOG(EXCEPTION) << "The tensor requires grad is false, which can not register tensor hook";
  }
  return grad_node;
}

inline uint64_t GetTensorNumId(const std::string &id) { return std::stoull(id.substr(1)); }
}  // namespace

PyTensorBackwardNodePreHook::PyTensorBackwardNodePreHook(const py::function &hook_fn, size_t output_idx)
    : hook_fn_(hook_fn), output_idx_(output_idx) {}

PyTensorBackwardNodePreHook::~PyTensorBackwardNodePreHook() {
  py::gil_scoped_acquire gil;
  hook_fn_ = py::object();
}

void PyTensorBackwardNodePreHook::operator()(ValuePtrList *grad) {
  if (output_idx_ >= grad->size()) {
    MS_LOG(EXCEPTION) << "PyTensor hook output_idx out of range";
  }

  py::gil_scoped_acquire gil;
  const auto py_grad = CValueToPybindObj((*grad)[output_idx_]);
  const auto ret = hook_fn_(py_grad);
  if (!ret.is_none()) {
    if (tensor::IsTensorPy(ret)) {
      (*grad)[output_idx_] = tensor::ConvertToTensor(ret);
    } else {
      MS_LOG(EXCEPTION) << "Tensor hook should be return Tensor, but get type: "
                        << py::str(ret.get_type().attr("__name__")).cast<std::string>() << ".";
    }
  }
}

uint64_t RegisterHook::RegisterTensorBackwardHook(const tensor::TensorPtr &tensor, const py::function &hook) {
  ++unique_id_;
  MS_LOG(DEBUG) << "Register hook " << py::str(py::cast<py::object>(hook)).cast<std::string>() << " for tensor "
                << tensor->id() << " with handle " << unique_id_;

  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    const auto &tensor_id = GetTensorNumId(tensor->id());
    std::shared_ptr<std::map<uint64_t, py::function>> hook_map;
    if (tensor->has_user_data("backward_hook")) {
      hook_map = tensor->user_data<std::map<uint64_t, py::function>>("backward_hook");
    } else {
      hook_map = std::make_shared<std::map<uint64_t, py::function>>();
      const_cast<tensor::TensorPtr &>(tensor)->set_user_data("backward_hook", hook_map);
    }
    (*hook_map)[unique_id_] = hook;

    unique_id_with_tensor_id_[unique_id_] = tensor_id;
    if (tensor_id_with_hook_map_.find(tensor_id) == tensor_id_with_hook_map_.end()) {
      tensor_id_with_hook_map_[tensor_id] = hook_map;
    }
  } else {
    auto grad_node = BuildAutoGradMeta(tensor);
    grad_node->AddPyTensorHook(
      unique_id_, std::make_unique<PyTensorBackwardNodePreHook>(hook, tensor->auto_grad_meta_data()->output_index()));
    hook_id_node_map_.emplace(unique_id_, grad_node);
  }
  return unique_id_;
}

void RegisterHook::RemoveTensorBackwardHookOfGraph(uint64_t tensor_id, uint64_t handle_id) {
  auto found = tensor_id_with_hook_map_.find(tensor_id);
  if (found != tensor_id_with_hook_map_.end()) {
    auto hook_map = found->second.lock();
    if (hook_map != nullptr) {
      auto iter = hook_map->find(handle_id);
      if (iter != hook_map->end()) {
        MS_LOG(DEBUG) << "Remove hook, handle id: " << handle_id
                      << ", hook: " << py::cast<std::string>(py::str(iter->second));
        hook_map->erase(iter);
      } else {
        MS_LOG(WARNING) << "No hook was found for handle id: " << handle_id;
      }
    }
  }
}

void RegisterHook::RemoveTensorBackwardHook(uint64_t handle_id) {
  MS_LOG(DEBUG) << "Remove hook by id " << handle_id;
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (const auto iter = unique_id_with_tensor_id_.find(handle_id); iter != unique_id_with_tensor_id_.end()) {
      RemoveTensorBackwardHookOfGraph(iter->second, handle_id);
      unique_id_with_tensor_id_.erase(iter);
    }
  } else {
    // For inplace ops, the grad_node might not have been updated yet at this point.
    // So it is necessary to wait frontend and bprop stage.
    runtime::Pipeline::Get().WaitFrontend();
    runtime::Pipeline::Get().WaitBpropStage();
    if (const auto iter = hook_id_node_map_.find(handle_id); iter != hook_id_node_map_.end()) {
      if (auto grad_node = iter->second.lock(); grad_node != nullptr) {
        grad_node->RemovePyTensorHook(handle_id);
      }
      hook_id_node_map_.erase(iter);
    }
  }
}

py::list RegisterHook::GetHooks(const tensor::TensorPtr &tensor) {
  py::list hooks;
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    const auto &tensor_id = GetTensorNumId(tensor->id());
    auto found = tensor_id_with_hook_map_.find(tensor_id);
    if (found != tensor_id_with_hook_map_.end()) {
      auto hook_map = found->second.lock();
      if (hook_map != nullptr) {
        for (const auto &item : *hook_map) {
          hooks.append(item.second);
        }
      }
    }
  } else {
    runtime::Pipeline::Get().WaitFrontend();
    runtime::Pipeline::Get().WaitBpropStage();
    if (const auto auto_grad_meta_data = impl::GetAutogradMetaImpl(tensor)) {
      const auto output_idx = auto_grad_meta_data->output_index();
      if (const auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl()) {
        const auto &py_tensor_pre_hooks = grad_node->py_tensor_pre_hooks();
        for (const auto &item : py_tensor_pre_hooks) {
          const auto &py_hooks = item.second;
          if (py_hooks->output_idx_ == output_idx) {
            hooks.append(py_hooks->hook_fn_);
          }
        }
      }
    }
  }
  return hooks;
}

struct HookAdapterRegister {
  HookAdapterRegister() {
    MS_LOG(DEBUG) << "Register hook adapter";
    HookAdapter::SetRegisterTensorBackwardHookHandler(
      [](const tensor::TensorPtr &tensor, const py::function &hook) -> uint64_t {
        return RegisterHook::RegisterTensorBackwardHook(tensor, hook);
      });

    HookAdapter::SetRemoveTensorBackwardHookHandler(
      [](uint64_t id) -> void { RegisterHook::RemoveTensorBackwardHook(id); });

    HookAdapter::SetGetHooksHandler(
      [](const tensor::TensorPtr &tensor) -> py::list { return RegisterHook::GetHooks(tensor); });
  }
} hook_adapter_register;
}  // namespace mindspore::pynative::autograd
