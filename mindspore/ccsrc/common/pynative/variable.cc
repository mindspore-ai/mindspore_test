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

#include "include/common/pynative//variable.h"
#include <memory>
#include "include/common/pynative/common_utils.h"
#include "debug/profiler/profiler.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"
#include "mindspore/core/include/utils/ms_context.h"

namespace mindspore::pynative::autograd {
namespace {
uint64_t AddId() {
  static uint64_t id = 0;
  return id++;
}

void RebuildTensorGrad(const BackwardNodePtr &grad_node, const tensor::TensorPtr &tensor, size_t output_index) {
  MS_EXCEPTION_IF_NULL(grad_node);
  auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>(grad_node, InputType::kOpOutput);
  auto_grad_meta_data->set_output_index(output_index);
  tensor->set_auto_grad_meta_data(auto_grad_meta_data);
}

bool IsOutputPlaceHolder(const ValuePtr &output) {
  if (output->isa<tensor::Tensor>()) {
    return !output->cast<tensor::TensorPtr>()->used_in_bprop_graph();
  }
  if (output->isa<ValueSequence>()) {
    const auto &val_seq = output->cast<ValueSequencePtr>()->value();
    return std::all_of(val_seq.begin(), val_seq.end(), [](const auto &val) { return IsOutputPlaceHolder(val); });
  }
  return true;
}

AutoDiffInterfacePtr local_auto_diff_engine{nullptr};
}  // namespace

ValuePtr SavedNode::Unwrap(BackwardNodePtr grad_node, bool only_tensor) {
  if (is_placeholder_) {
    return data_;
  }
  auto copy_value = CommonUtils::ShallowCopyAndDetach(data_);
  ValuePtrList flatten_outputs;
  if (!only_tensor) {
    flatten_outputs = CommonUtils::FlattenTensorSeqInValue(copy_value);
  } else {
    flatten_outputs = CommonUtils::FlattenOnlyTensor(copy_value);
  }
  if (flatten_outputs.size() != grad_node->output_size()) {
    MS_LOG(EXCEPTION) << "flatten outputs size should be same as grad_node output size, but got: "
                      << flatten_outputs.size() << " vs " << grad_node->output_size();
  }
  if (is_view_inplace_) {
    grad_node = weak_grad_node_.lock();
  }
  for (size_t i = 0; i < flatten_outputs.size(); ++i) {
    auto tensor = flatten_outputs[i]->cast<tensor::TensorPtr>();
    if (tensor != nullptr) {
      RebuildTensorGrad(grad_node, tensor, i);
    }
  }
  return copy_value;
}

SavedNodePtr SavedNode::ConstructSavedNode(const ValuePtr &output, bool is_view_inplace) {
  MS_EXCEPTION_IF_NULL(output);
  bool is_placeholder = IsOutputPlaceHolder(output);
  auto detach_value = CommonUtils::ShallowCopyAndDetach(output);
  if (is_view_inplace) {
    auto tensor = output->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto grad_node = impl::GetUnsafeGradNodeImpl(tensor);
    MS_EXCEPTION_IF_NULL(grad_node);
    return std::make_shared<SavedNode>(detach_value, grad_node, true, is_placeholder);
  }
  return std::make_shared<SavedNode>(detach_value, nullptr, false, is_placeholder);
}

BackwardNode::BackwardNode(string name, uint64_t seq_id, size_t output_size) noexcept
    : name_(std::move(name)), seq_id_(seq_id), output_size_(output_size) {}

BackwardNode::BackwardNode(string name, size_t output_size) noexcept
    : BackwardNode(std::move(name), AddId(), output_size) {}

ValuePtrList BackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  auto flatten_gradients = CommonUtils::FlattenTensorSeqInValueSeq(gradient_value, false);
  return flatten_gradients;
}

bool BackwardNode::IsEmpty() {
  if (std::all_of(next_edges().begin(), next_edges().end(),
                  [](const Edge &edge) -> bool { return !edge.is_defined(); })) {
    return true;
  }
  return false;
}

std::string BackwardNode::ToString() const {
  std::ostringstream buf;
  buf << "Node name: " << name() << "\n";
  for (size_t i = 0; i < next_edges().size(); ++i) {
    if (!next_edges()[i].is_defined()) {
      buf << "Last edge: " << i << " undefined edge"
          << "\n";
      continue;
    }
    const auto &last_grad_node = next_edges()[i].grad_node;
    auto index = next_edges()[i].input_index;
    buf << "Last edge: " << i << ", node name: " << last_grad_node->name() << ", output index: " << index << "\n";
  }
  return buf.str();
}

AutoDiffGuard::AutoDiffGuard(const AutoDiffInterfacePtr &auto_diff) {
  prev_auto_diff_engine_ = local_auto_diff_engine;
  local_auto_diff_engine = auto_diff;
}

AutoDiffGuard::~AutoDiffGuard() {
  local_auto_diff_engine = prev_auto_diff_engine_;
  prev_auto_diff_engine_ = nullptr;
}

namespace impl {
AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return GetAutogradMetaImpl(*tensor);
}

AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::Tensor &tensor) {
  auto auto_grad_meta = tensor.auto_grad_meta_data();
  if (auto_grad_meta == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<AutoGradMetaData>(auto_grad_meta);
}

ViewAutoGradMetaDataPtr GetViewAutogradMetaImpl(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->auto_grad_meta_data() == nullptr) {
    return nullptr;
  }
  const auto &meta_data = tensor->auto_grad_meta_data();
  auto view_meta_data = std::dynamic_pointer_cast<ViewAutoGradMetaData>(meta_data);
  return view_meta_data;
}

BackwardNodePtr GetUnsafeGradNodeImpl(const tensor::TensorPtr &tensor) {
  if (tensor->auto_grad_meta_data() != nullptr) {
    return tensor->auto_grad_meta_data()->UnsafeGetGradNodeImpl();
  }
  return nullptr;
}

bool RequiresGrad(const tensor::TensorPtr &tensor) {
  auto grad_node = GetUnsafeGradNodeImpl(tensor);
  if (local_auto_diff_engine == nullptr) {
    return grad_node != nullptr;
  } else {
    return grad_node != nullptr && local_auto_diff_engine->IsInExecGraph(grad_node);
  }
}

AutoDiffInterfacePtr CurrentAutoDiffEngine() { return local_auto_diff_engine; }
}  // namespace impl
}  // namespace mindspore::pynative::autograd
