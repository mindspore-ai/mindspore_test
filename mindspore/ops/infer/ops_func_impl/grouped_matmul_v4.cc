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
void GroupedMatmulV4FuncImpl::FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  // for tensortuple(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < kIndex12; i++) {
    const auto &tensors = input_infos[i];
    if (tensors->IsNone()) {
      dyn_input_sizes.push_back(0);
      continue;
    }
    if (i == LongToSize(idxes_.group_list)) {
      dyn_input_sizes.push_back(1);
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

int64_t GroupedMatmulV4FuncImpl::FetchGroupListSize(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  const auto &group_list_shape = input_infos[idxes_.group_list]->GetShape();
  MS_CHECK_VALUE(group_list_shape.size() == kIndex1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's rank", group_list_shape.size(), kEqual,
                                                             kIndex1, primitive));
  return input_infos[idxes_.group_list]->IsDynamic() ? abstract::Shape::kShapeDimAny : group_list_shape[kIndex0];
}

TypeIdList GroupedMatmulV4FuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->IsEnableInferBoost() && ms_context->ascend_soc_version() == kAscendVersion310p) {
    return {kNumberTypeFloat16};
  }

  const auto &x_tensors = input_infos[idxes_.x]->GetSequenceElements();
  const auto &per_token_scale_info = input_infos[per_token_scale_idx_];
  TypeIdList output_types;
  if (per_token_scale_info->IsNone()) {
    std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                   [](const InferInfoPtr &info) { return info->GetType(); });
  } else {
    if (input_infos[scale_idx_]->IsNone()) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the scale cannot be None in per-token quant.";
    }
    const auto &scale_tensors = input_infos[scale_idx_]->GetSequenceElements();
    for (const InferInfoPtr &info : scale_tensors) {
      auto scale_type = info->GetType();
      if (scale_type != kNumberTypeBFloat16) {
        MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', the scales must be BFloat16 in per-token quant.";
      }
      output_types.emplace_back(scale_type);
    }
  }
  return output_types;
}
}  // namespace ops
}  // namespace mindspore
