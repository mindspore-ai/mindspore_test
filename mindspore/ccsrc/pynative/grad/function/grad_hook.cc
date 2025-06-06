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

#include "pynative/grad/function/grad_hook.h"
#include "pynative/grad/function/func_grad.h"
#include "mindspore/core/include/ir/tensor.h"
#include <memory>
#include <string>

namespace mindspore::pynative::autograd {
namespace {
class GradHookRegister {
 public:
  GradHookRegister() { tensor::Tensor::InitilizeGradImpl(std::make_unique<GradHook>()); }
};
[[maybe_unused]] GradHookRegister grad_hook_register;
}  // namespace

BackwardNodePtr GradHook::grad_node(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  return SafeGetGradNodeImpl(self);
}

TensorPtr GradHook::grad(const TensorPtr &self) const {
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    return nullptr;
  }
  return grad_meta->grad();
}

void GradHook::set_grad(const TensorPtr &self, const TensorPtr &grad) {
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    grad_meta = std::make_shared<AutoGradMetaData>();
    self->set_auto_grad_meta_data(grad_meta);
  }
  grad_meta->set_grad(grad);
}

bool GradHook::is_leaf(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  const auto &grad_node = impl::GetUnsafeGradNodeImpl(self);
  if (grad_node == nullptr || grad_node->IsLeaf()) {
    return true;
  }
  return false;
}

bool GradHook::requires_grad(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    return false;
  }
  return self->auto_grad_meta_data()->requires_grad();
}

void GradHook::set_requires_grad(const TensorPtr &self, bool requires_grad) {
  runtime::Pipeline::Get().WaitBpropStage();
  if (!is_leaf(self) && !requires_grad) {
    MS_LOG(EXCEPTION) << "You can only set requires_grad false to a leaf tensor! You can use tensor.detach() to "
                         "set output tensor to calculate grad.";
  }
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    if (!requires_grad) {
      return;
    }
    grad_meta = std::make_shared<AutoGradMetaData>();
    self->set_auto_grad_meta_data(grad_meta);
  }
  grad_meta->set_requires_grad(requires_grad);
  if (impl::GetUnsafeGradNodeImpl(self) == nullptr && requires_grad) {
    grad_meta->set_grad_node(std::make_shared<autograd::LeafNode>(
      self->param_info() != nullptr ? self->param_info()->name() : "input_" + self->id(), self, self->shape(),
      self->Dtype(), self->is_parameter()));
    grad_meta->set_input_type(self->is_parameter() ? InputType::kParameter : InputType::kInput);
  }
}
}  // namespace mindspore::pynative::autograd
