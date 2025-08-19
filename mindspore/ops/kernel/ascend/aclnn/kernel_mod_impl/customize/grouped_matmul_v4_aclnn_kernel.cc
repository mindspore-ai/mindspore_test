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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/grouped_matmul_v4_aclnn_kernel.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace grouped_matmul_v4 {
namespace {
std::vector<std::vector<KernelTensor *>> DealWithListTensors(const std::vector<int64_t> &group_info,
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

std::vector<int64_t> ComputeStartIdxsFromGroupInfo(const std::vector<int64_t> &group_info) {
  std::vector<int64_t> start_idxs{0};
  int64_t cur_end_idx = 0;
  for (size_t i = 0; i < group_info.size(); ++i) {
    cur_end_idx += (group_info[i] == 0 ? 1 : group_info[i]);
    start_idxs.push_back(cur_end_idx);
  }
  return start_idxs;
}
}  // namespace

static inline void UnifyWeightShape(const std::vector<KernelTensor *> &ori_weights,
                                    std::vector<std::shared_ptr<KernelTensor>> *new_weights_shared_ptr,
                                    std::vector<KernelTensor *> *new_weights_raw_ptr) {
  for (const auto &w : ori_weights) {
    if (w->dtype_id() == kNumberTypeInt4) {
      const auto &storage_info = w->tensor_storage_info();
      if (storage_info != nullptr && !storage_info->is_contiguous) {
        MS_LOG(EXCEPTION) << "Currently, GroupedMatMulV4 does not support noncontiguous input tensor for int4 quant, "
                          << "but got noncontiguous input tensor: " << w->ToString()
                          << ", storage info: " << storage_info->ToString();
      }
      auto new_w = w->CloneKernelTensor();
      auto w_shape = w->GetShapeVector();
      w_shape.back() *= kSizeTwo;
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
  start_idxs_ = ComputeStartIdxsFromGroupInfo(group_info_);

  auto list_inputs = DealWithListTensors(group_info_, start_idxs_, inputs);
  const auto &group_list = inputs[start_idxs_[group_list_idx_]];
  const auto split_item_idx = start_idxs_.back();
  split_item_ = inputs.at(split_item_idx)->GetValueWithCheck<int64_t>();
  group_type_ = inputs.at(split_item_idx + kIndex1)->GetValueWithCheck<int64_t>();
  group_list_type_ = inputs.at(split_item_idx + kIndex2)->GetValueWithCheck<int64_t>();
  act_type_ = inputs.at(split_item_idx + kIndex3)->GetValueWithCheck<int64_t>();

  std::vector<std::shared_ptr<KernelTensor>> new_weights;
  std::vector<KernelTensor *> new_weights_raw;
  UnifyWeightShape(list_inputs[kIndex1], &new_weights, &new_weights_raw);

  GetWorkspaceForResize(list_inputs[kIndex0], new_weights_raw, list_inputs[kIndex2], list_inputs[kIndex3],
                        list_inputs[kIndex4], list_inputs[kIndex5], list_inputs[kIndex6], list_inputs[kIndex7],
                        group_list, list_inputs[kIndex9], list_inputs[kIndex10], list_inputs[kIndex11], split_item_,
                        group_type_, group_list_type_, act_type_, outputs, activation_feature_out_,
                        dyn_quant_scale_out_);
}

bool GroupedMatmulV4Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto list_inputs = DealWithListTensors(group_info_, start_idxs_, inputs);
  const auto &group_list = inputs[start_idxs_[group_list_idx_]];

  std::vector<std::shared_ptr<KernelTensor>> new_weights;
  std::vector<KernelTensor *> new_weights_raw;
  UnifyWeightShape(list_inputs[kIndex1], &new_weights, &new_weights_raw);

  RunOp(stream_ptr, workspace, list_inputs[kIndex0], new_weights_raw, list_inputs[kIndex2], list_inputs[kIndex3],
        list_inputs[kIndex4], list_inputs[kIndex5], list_inputs[kIndex6], list_inputs[kIndex7], group_list,
        list_inputs[kIndex9], list_inputs[kIndex10], list_inputs[kIndex11], split_item_, group_type_, group_list_type_,
        act_type_, outputs, activation_feature_out_, dyn_quant_scale_out_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GroupedMatmulV4, GroupedMatmulV4Ascend);
}  // namespace grouped_matmul_v4
}  // namespace kernel
}  // namespace mindspore
