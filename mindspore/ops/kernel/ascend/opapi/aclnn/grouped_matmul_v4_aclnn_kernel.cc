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
#include "kernel/ascend/opapi/aclnn/grouped_matmul_v4_aclnn_kernel.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/kernel.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputWeightIdx = 1;
constexpr size_t kInputBiasIdx = 2;
constexpr size_t kInputScaleIdx = 3;
constexpr size_t kInputOffsetIdx = 4;
constexpr size_t kInputAntiquantScaleIdx = 5;
constexpr size_t kInputAntiquantOffsetIdx = 6;
constexpr size_t kInputPreTokenScaleIdx = 7;
constexpr size_t kInputGroupListIdx = 8;
constexpr size_t kInputActivationInputIdx = 9;
constexpr size_t kInputActivationQuantScaleIdx = 10;
constexpr size_t kInputActivationQuantOffsetIdx = 11;

std::vector<std::vector<KernelTensor *>> DealWithGroupedMatmulListTensors(const std::vector<int64_t> &group_info,
                                                                          const std::vector<int64_t> &start_idxs,
                                                                          const std::vector<KernelTensor *> &inputs) {
  // x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pre_token_scale, activation_input,
  // activation_quant_scale, activation_quant_offset would be list[tensor] or None
  std::vector<std::vector<KernelTensor *>> list_inputs{};
  for (size_t i = 0; i < kIndex12; i++) {
    std::vector<KernelTensor *> input_i{};
    if (group_info[i] > 0) {
      input_i.assign(inputs.begin() + start_idxs[i], inputs.begin() + start_idxs[i + 1]);
    }
    (void)list_inputs.emplace_back(std::move(input_i));
  }
  return list_inputs;
}
}  // namespace

static inline void UnifyWeightShape(const std::vector<KernelTensor *> &ori_weights,
                                    std::vector<std::shared_ptr<KernelTensor>> *new_weights_shared_ptr,
                                    std::vector<KernelTensor *> *new_weights_raw_ptr) {
  for (const auto &w : ori_weights) {
    if (w->dtype_id() == kNumberTypeInt4) {
      auto new_w = std::make_shared<KernelTensor>(*w);
      auto w_shape = w->GetShapeVector();
      w_shape.back() *= 2;
      new_w->SetShapeVector(w_shape);
      new_weights_shared_ptr->emplace_back(new_w);
      new_weights_raw_ptr->emplace_back(new_w.get());
    } else {
      new_weights_raw_ptr->emplace_back(w);
    }
  }
}

void GroupedMatmulV4Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  group_info_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("group_info"));
  start_idxs_.clear();
  (void)start_idxs_.emplace_back(0);
  int64_t cur_end_idx = 0;
  for (size_t i = 0; i < kIndex12; ++i) {
    cur_end_idx += (group_info_[i] == 0 ? 1 : group_info_[i]);
    (void)start_idxs_.emplace_back(cur_end_idx);
  }

  auto list_inputs = DealWithGroupedMatmulListTensors(group_info_, start_idxs_, inputs);
  auto group_list_tensor = *(inputs.begin() + start_idxs_[kInputGroupListIdx]);

  auto split_item_tensor = inputs.at(inputs.size() - kIndex4);
  MS_EXCEPTION_IF_NULL(split_item_tensor);
  split_item_ = split_item_tensor->GetValueWithCheck<int64_t>();

  auto group_type_tensor = inputs.at(inputs.size() - kIndex3);
  MS_EXCEPTION_IF_NULL(group_type_tensor);
  group_type_ = group_type_tensor->GetValueWithCheck<int64_t>();

  auto group_list_type_tensor = inputs.at(inputs.size() - kIndex2);
  MS_EXCEPTION_IF_NULL(group_list_type_tensor);
  group_list_type_ = group_list_type_tensor->GetValueWithCheck<int64_t>();

  auto act_type_tensor = inputs.at(inputs.size() - kIndex1);
  MS_EXCEPTION_IF_NULL(act_type_tensor);
  act_type_ = act_type_tensor->GetValueWithCheck<int64_t>();

  std::vector<std::shared_ptr<KernelTensor>> new_weights;
  std::vector<KernelTensor *> new_weights_raw;
  UnifyWeightShape(list_inputs[kInputWeightIdx], &new_weights, &new_weights_raw);

  GetWorkspaceForResize(list_inputs[kInputXIdx], new_weights_raw, list_inputs[kInputBiasIdx],
                        list_inputs[kInputScaleIdx], list_inputs[kInputOffsetIdx], list_inputs[kInputAntiquantScaleIdx],
                        list_inputs[kInputAntiquantOffsetIdx], list_inputs[kInputPreTokenScaleIdx], group_list_tensor,
                        list_inputs[kInputActivationInputIdx], list_inputs[kInputActivationQuantScaleIdx],
                        list_inputs[kInputActivationQuantOffsetIdx], split_item_, group_type_, group_list_type_,
                        act_type_, outputs, nullptr, nullptr);
}

bool GroupedMatmulV4Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto list_inputs = DealWithGroupedMatmulListTensors(group_info_, start_idxs_, inputs);
  auto group_list_tensor = *(inputs.begin() + start_idxs_[kInputGroupListIdx]);
  std::vector<std::shared_ptr<KernelTensor>> new_weights;
  std::vector<KernelTensor *> new_weights_raw;
  UnifyWeightShape(list_inputs[kInputWeightIdx], &new_weights, &new_weights_raw);

  RunOp(stream_ptr, workspace, list_inputs[kInputXIdx], new_weights_raw, list_inputs[kInputBiasIdx],
        list_inputs[kInputScaleIdx], list_inputs[kInputOffsetIdx], list_inputs[kInputAntiquantScaleIdx],
        list_inputs[kInputAntiquantOffsetIdx], list_inputs[kInputPreTokenScaleIdx], group_list_tensor,
        list_inputs[kInputActivationInputIdx], list_inputs[kInputActivationQuantScaleIdx],
        list_inputs[kInputActivationQuantOffsetIdx], split_item_, group_type_, group_list_type_, act_type_, outputs,
        nullptr, nullptr);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GroupedMatmulV4, GroupedMatmulV4Ascend);
}  // namespace kernel
}  // namespace mindspore
