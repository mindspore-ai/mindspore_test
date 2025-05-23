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
#include "infer/ops_func_impl/grouped_matmul_v4.h"

#include <vector>
#include <algorithm>

#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
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
void GroupedMatmulV4FuncImpl::FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  // for tensortuple(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < kIndex12; i++) {
    const auto &tensors = input_infos[i];
    if (i == LongToSize(group_list_idx_)) {
      dyn_input_sizes.push_back(1);
      continue;
    }
    if (tensors->IsNone()) {
      dyn_input_sizes.push_back(0);
      continue;
    }
    if (MS_UNLIKELY(tensors->IsDynamicSequence())) {
      MS_EXCEPTION(RuntimeError)
        << "For '" << primitive->name()
        << "', all inputs which is list[tensor] should not be dynamic sequence, which is not supported.";
    }
    const auto &elements = tensors->GetSequenceElements();
    dyn_input_sizes.push_back(SizeToLong(elements.size()));
  }
  primitive->set_attr("group_info", MakeValue(dyn_input_sizes));  // len of tuple input
}

int64_t GroupedMatmulV4FuncImpl::FetchGroupListIndex(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  if (MS_LIKELY(input_infos[idxes_.x]->IsSequence())) {
    return group_list_idx_;
  }
  // Runtime phase: the element in input_args is KernelTensor. (tuple is expanded)
  const auto &group_info = GetValue<std::vector<int64_t>>(primitive->GetAttr("group_info"));
  const auto &start_idxes = ComputeStartIdxsFromGroupInfo(group_info);
  return start_idxes.at(group_list_idx_);
}

int64_t GroupedMatmulV4FuncImpl::FetchGroupListSize(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  const auto group_list_idx = FetchGroupListIndex(primitive, input_infos);
  const auto &group_list_shape = input_infos.at(group_list_idx)->GetShape();
  MS_CHECK_VALUE(group_list_shape.size() == kIndex1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's rank", group_list_shape.size(), kEqual,
                                                             kIndex1, primitive));
  return input_infos[group_list_idx]->IsDynamic() ? abstract::Shape::kShapeDimAny : group_list_shape[kIndex0];
}

TypeIdList GroupedMatmulV4FuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto &x_tensors = input_infos[idxes_.x]->GetSequenceElements();
  const auto &scale_infos = input_infos[scale_idx_];
  TypeIdList output_types;
  if (scale_infos->IsNone()) {
    std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                   [](const InferInfoPtr &info) { return info->GetType(); });
  } else {
    const auto &scale_tensors = scale_infos->GetSequenceElements();
    TypeId scale_type = scale_tensors[0]->GetType();
    if (scale_type == kNumberTypeUInt64) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeInt8; });
    } else if (scale_type == kNumberTypeBFloat16) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeBFloat16; });
    } else if (scale_type == kNumberTypeFloat32) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeFloat16; });
    } else {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the scale only support Uint16, BFloat16 and Float32, but got " << scale_type;
    }
  }
  return output_types;
}
}  // namespace ops
}  // namespace mindspore
