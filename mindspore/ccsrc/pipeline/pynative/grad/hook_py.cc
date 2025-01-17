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

#include "pipeline/pynative/grad/hook_py.h"
#include <memory>
#include <string>
#include "include/common/utils/hook.h"
#include "pipeline/jit/ps/pipeline.h"

namespace mindspore::pynative::autograd {
namespace {
AutoGradMetaDataWeakPtr BuildAutoGradMeta(const tensor::Tensor &tensor) {
  auto auto_grad_meta_data = impl::get_autograd_meta_impl(tensor);
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    const_cast<tensor::Tensor &>(tensor).set_auto_grad_meta_data(auto_grad_meta_data);
    MS_LOG(DEBUG) << "Tensor has no auto_grad_meta_data, build it";
  }
  return {auto_grad_meta_data};
}

inline uint64_t GetTensorNumId(const std::string &id) { return std::stoull(id.substr(1)); }
}  // namespace

std::map<uint64_t, std::vector<uint64_t>> RegisterHook::tensor_id_with_unique_id_ = {};
std::map<uint64_t, std::weak_ptr<std::map<uint64_t, py::function>>> RegisterHook::tensor_id_with_hook_map_ = {};
std::map<uint64_t, std::pair<AutoGradMetaDataWeakPtr, TensorBackwardHookPtr>> RegisterHook::hook_meta_fn_map_ = {};

uint64_t RegisterHook::RegisterTensorBackwardHook(const tensor::Tensor &tensor, const py::function &hook) {
  // Delete char 'T'
  const auto &tensor_id = GetTensorNumId(tensor.id());
  ++unique_id_;
  MS_LOG(DEBUG) << "Register hook " << py::str(py::cast<py::object>(hook)).cast<std::string>() << " for tensor "
                << tensor.id() << " with handle " << unique_id_;

  // Add hook for tensor
  auto meta = BuildAutoGradMeta(tensor);
  auto tensor_backward_hook = std::make_shared<TensorBackwardHook>(tensor_id, hook);
  MS_EXCEPTION_IF_NULL(meta.lock());
  // If tensor has register hook before and finish once grad; And then register another hook fn, auto grad meta is not
  // nullptr and UpdateTensorBackwardHook will not be call at PyNative forward process. so Call it here.
  UpdateTensorBackwardHook(meta.lock(), tensor.id());
  meta.lock()->AddBackwardHook(unique_id_, tensor_backward_hook);
  hook_meta_fn_map_.emplace(unique_id_, std::make_pair(meta, tensor_backward_hook));
  tensor_id_with_unique_id_[tensor_id].emplace_back(unique_id_);

  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    std::shared_ptr<std::map<uint64_t, py::function>> hook_map;
    if (tensor.has_user_data("backward_hook")) {
      hook_map = tensor.user_data<std::map<uint64_t, py::function>>("backward_hook");
    } else {
      hook_map = std::make_shared<std::map<uint64_t, py::function>>();
      const_cast<tensor::Tensor &>(tensor).set_user_data("backward_hook", hook_map);
    }
    (*hook_map)[unique_id_] = hook;

    if (tensor_id_with_hook_map_.find(tensor_id) == tensor_id_with_hook_map_.end()) {
      tensor_id_with_hook_map_[tensor_id] = hook_map;
    }
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
  const auto it = hook_meta_fn_map_.find(handle_id);
  if (it == hook_meta_fn_map_.end()) {
    MS_LOG(DEBUG) << "Can not find in hook meta fn map";
    return;
  }
  for (auto tensor_it = tensor_id_with_unique_id_.begin(); tensor_it != tensor_id_with_unique_id_.end();) {
    auto tensor_id = tensor_it->first;
    auto &unique_id_list = tensor_it->second;
    auto new_end = std::remove(unique_id_list.begin(), unique_id_list.end(), handle_id);
    if (new_end != unique_id_list.end()) {
      unique_id_list.erase(new_end, unique_id_list.end());
      if (unique_id_list.empty()) {
        tensor_it = tensor_id_with_unique_id_.erase(tensor_it);
      }

      if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
        RemoveTensorBackwardHookOfGraph(tensor_id, handle_id);
      }
      break;
    } else {
      ++tensor_it;
    }
  }
  auto meta = it->second.first.lock();
  (void)hook_meta_fn_map_.erase(it);
  if (meta == nullptr) {
    MS_LOG(DEBUG) << "Get null meta";
    return;
  }
  meta->RemoveBackwardHook(handle_id);
}

py::list RegisterHook::GetHooks(const tensor::Tensor &tensor) {
  const auto &tensor_id = GetTensorNumId(tensor.id());
  py::list hooks;

  auto found = tensor_id_with_hook_map_.find(tensor_id);
  if (found != tensor_id_with_hook_map_.end()) {
    auto hook_map = found->second.lock();
    if (hook_map != nullptr) {
      for (const auto &item : *hook_map) {
        hooks.append(item.second);
      }
    }
  }

  return hooks;
}

void RegisterHook::UpdateTensorBackwardHook(const AutoGradMetaDataPtr &auto_grad_meta_data,
                                            const std::string &tensor_id) {
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  const auto &tensor_numerical_id = GetTensorNumId(tensor_id);
  auto it = tensor_id_with_unique_id_.find(tensor_numerical_id);
  if (it == tensor_id_with_unique_id_.end()) {
    return;
  }
  MS_LOG(DEBUG) << "Update tensor backward hook for tensor id " << tensor_id;
  for (uint64_t unique_id : tensor_id_with_unique_id_[tensor_numerical_id]) {
    auto fn_it = hook_meta_fn_map_.find(unique_id);
    if (fn_it != hook_meta_fn_map_.end()) {
      auto_grad_meta_data->AddBackwardHook(unique_id, fn_it->second.second);
      // Update remove handle auto grad meta
      hook_meta_fn_map_[unique_id].first = std::weak_ptr<AutoGradMetaData>(auto_grad_meta_data);
    }
  }
}
}  // namespace mindspore::pynative::autograd
