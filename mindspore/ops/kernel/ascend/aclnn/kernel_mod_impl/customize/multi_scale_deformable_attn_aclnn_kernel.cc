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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/multi_scale_deformable_attn_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace multi_scale_deformable_attn {
void MultiScaleDeformableAttnAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  expand_indices_.clear();

  auto dtype_id = inputs[kIndex0]->dtype_id();
  if (dtype_id != kNumberTypeFloat32 && dtype_id != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For MSDA, value tensor type " << inputs[kIndex0]->GetType() << " is illegal";
  }

  dtype_id = inputs[kIndex1]->dtype_id();
  if (dtype_id != kNumberTypeInt32 && dtype_id != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For MSDA, shape tensor type " << inputs[kIndex1]->GetType() << " is illegal";
  }

  dtype_id = inputs[kIndex2]->dtype_id();
  if (dtype_id != kNumberTypeInt32 && dtype_id != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For MSDA, offset tensor type " << inputs[kIndex2]->GetType() << " is illegal";
  }

  dtype_id = inputs[kIndex3]->dtype_id();
  if (dtype_id != kNumberTypeFloat32 && dtype_id != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For MSDA, locations tensor type " << inputs[kIndex3]->GetType() << " is illegal";
  }

  dtype_id = inputs[kIndex4]->dtype_id();
  if (dtype_id != kNumberTypeFloat32 && dtype_id != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For MSDA, weight tensor type " << inputs[kIndex3]->GetType() << " is illegal";
  }

  const auto value_shape = inputs[kIndex0]->GetShapeVector();
  value_expand_.SetType(std::make_shared<TensorType>(kFloat32));
  value_expand_.SetShape(std::make_shared<abstract::TensorShape>(value_shape));
  auto value = &value_expand_;
  GetWorkspaceForResizeCastValue(inputs[kIndex0], kFloat32, &value_expand_);
  expand_indices_.emplace_back(kIndex0);

  const auto shape_shape = inputs[kIndex1]->GetShapeVector();
  shape_expand_.SetType(std::make_shared<TensorType>(kInt32));
  shape_expand_.SetShape(std::make_shared<abstract::TensorShape>(shape_shape));
  auto shape = &shape_expand_;
  GetWorkspaceForResizeCastShape(inputs[kIndex1], kInt32, &shape_expand_);
  expand_indices_.emplace_back(kIndex1);

  const auto offset_shape = inputs[kIndex2]->GetShapeVector();
  offset_expand_.SetType(std::make_shared<TensorType>(kInt32));
  offset_expand_.SetShape(std::make_shared<abstract::TensorShape>(offset_shape));
  auto offset = &offset_expand_;
  GetWorkspaceForResizeCastOffset(inputs[kIndex2], kInt32, &offset_expand_);
  expand_indices_.emplace_back(kIndex2);

  const auto locations_shape = inputs[kIndex3]->GetShapeVector();
  locations_expand_.SetType(std::make_shared<TensorType>(kFloat32));
  locations_expand_.SetShape(std::make_shared<abstract::TensorShape>(locations_shape));
  auto locations = &locations_expand_;
  GetWorkspaceForResizeCastLocations(inputs[kIndex3], kFloat32, &locations_expand_);
  expand_indices_.emplace_back(kIndex3);

  const auto weight_shape = inputs[kIndex4]->GetShapeVector();
  weight_expand_.SetType(std::make_shared<TensorType>(kFloat32));
  weight_expand_.SetShape(std::make_shared<abstract::TensorShape>(weight_shape));
  auto weight = &weight_expand_;
  GetWorkspaceForResizeCastWeight(inputs[kIndex4], kFloat32, &weight_expand_);
  expand_indices_.emplace_back(kIndex4);

  const auto output_shape = outputs[kIndex0]->GetShapeVector();
  output_mid_.SetType(std::make_shared<TensorType>(kFloat32));
  output_mid_.SetShape(std::make_shared<abstract::TensorShape>(output_shape));
  const auto output_mid = &output_mid_;
  GetWorkspaceForResizeMultiScaleDeformableAttn(value, shape, offset, locations, weight, output_mid);
  expand_indices_.emplace_back(kIndex5);

  // add workspace
  const auto &value_output_size =
    ops::CalOutputSize(value->GetShapeVector(), mindspore::abstract::TypeIdSize(value->dtype_id()));
  workspace_size_list_.emplace_back(value_output_size);

  const auto &shape_output_size =
    ops::CalOutputSize(shape->GetShapeVector(), mindspore::abstract::TypeIdSize(shape->dtype_id()));
  workspace_size_list_.emplace_back(shape_output_size);

  const auto &offset_output_size =
    ops::CalOutputSize(offset->GetShapeVector(), mindspore::abstract::TypeIdSize(offset->dtype_id()));
  workspace_size_list_.emplace_back(offset_output_size);

  const auto &locations_output_size =
    ops::CalOutputSize(locations->GetShapeVector(), mindspore::abstract::TypeIdSize(locations->dtype_id()));
  workspace_size_list_.emplace_back(locations_output_size);

  const auto &weight_output_size =
    ops::CalOutputSize(weight->GetShapeVector(), mindspore::abstract::TypeIdSize(weight->dtype_id()));
  workspace_size_list_.emplace_back(weight_output_size);

  // add extra output_mid workspace
  const auto &output_mid_size =
    ops::CalOutputSize(outputs[kIndex0]->GetShapeVector(), mindspore::abstract::TypeIdSize(kNumberTypeFloat32));
  workspace_size_list_.emplace_back(output_mid_size);
}

bool MultiScaleDeformableAttnAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto value = inputs[kIndex0];

  auto ori_type = kFloat32;
  if (value->dtype_id() == kNumberTypeFloat16) {
    ori_type = kFloat16;
  }

  size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex0;
  value_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  value = &value_expand_;
  RunOpCastValue(stream_ptr, workspace, inputs[kIndex0], kFloat32, &value_expand_);

  workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex1;
  shape_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto shape = &shape_expand_;
  RunOpCastShape(stream_ptr, workspace, inputs[kIndex1], kInt32, &shape_expand_);

  workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex2;
  offset_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto offset = &offset_expand_;
  RunOpCastOffset(stream_ptr, workspace, inputs[kIndex2], kInt32, &offset_expand_);

  workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex3;
  locations_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto locations = &locations_expand_;
  RunOpCastLocations(stream_ptr, workspace, inputs[kIndex3], kFloat32, &locations_expand_);

  workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex4;
  weight_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto weight = &weight_expand_;
  RunOpCastWeight(stream_ptr, workspace, inputs[kIndex4], kFloat32, &weight_expand_);

  workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + kIndex5;
  output_mid_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto output_mid = &output_mid_;
  RunOpMultiScaleDeformableAttn(stream_ptr, workspace, value, shape, offset, locations, weight, output_mid);

  RunOpCastOutput(stream_ptr, workspace, output_mid, ori_type, outputs[0]);

  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(MultiScaleDeformableAttn, MultiScaleDeformableAttnAscend);
}  // namespace multi_scale_deformable_attn
}  // namespace kernel
}  // namespace mindspore
