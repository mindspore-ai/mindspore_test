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

#include "infer/ops_func_impl/grouped_matmul_v2.h"

#include <cstdint>
#include <vector>
#include <algorithm>

#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
void GroupedMatmulV2FuncImpl::FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  // for tensortuple(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < kIndex7; ++i) {
    const auto &tensors = input_infos[i];
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

int64_t GroupedMatmulV2FuncImpl::FetchGroupListIndex(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  const auto input_num = SizeToLong(input_infos.size());
  return input_num + group_list_offset_;
}

int64_t GroupedMatmulV2FuncImpl::FetchGroupListSize(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  const auto group_list_idx = FetchGroupListIndex(primitive, input_infos);
  const auto &group_list_opt = input_infos.at(group_list_idx)->GetArrayValue<int64_t>();
  if (MS_UNLIKELY(!group_list_opt.has_value())) {
    return abstract::Shape::kShapeDimAny;
  }
  const auto &group_list = group_list_opt.value();
  return SizeToLong(group_list.size());
}

TypeIdList GroupedMatmulV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto &x_tensors = input_infos[idxes_.x]->GetSequenceElements();
  TypeIdList output_types;
  std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                 [](const InferInfoPtr &info) { return info->GetType(); });
  return output_types;
}

int32_t GroupedMatmulV2FuncImpl::PrivateCheckValidation(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos, int64_t group_type) const {
  if (group_type == -1) {
    return OP_CHECK_SUCCESS;
  }

  ShapeVector x_shape;
  if (input_infos[idxes_.x]->IsSequence()) {
    x_shape = input_infos[idxes_.x]->GetSequenceElements().at(kIndex0)->GetShape();
  } else {
    x_shape = input_infos[idxes_.x]->GetShape();
  }

  const auto group_list_idx = FetchGroupListIndex(primitive, input_infos);
  const auto &group_list_info = input_infos.at(group_list_idx);
  auto group_list_opt = group_list_info->GetArrayValue<int64_t>();
  if (MS_UNLIKELY(IsDynamic(x_shape) || !group_list_opt.has_value() || group_list_opt.value().HasUnknownValue())) {
    return OP_CHECK_RETRY;
  }

  auto expect_sum = group_type == 0 ? x_shape.front() : x_shape.back();
  const auto &group_list = group_list_opt.value().ToVector();
  for (size_t i = 0; i < group_list.size(); ++i) {
    if (i == kIndex0) {
      MS_CHECK_VALUE(group_list[i] >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "first element of group_list", group_list[i], kGreaterEqual, 0, primitive));
    } else {
      if (MS_UNLIKELY(group_list[i] < group_list[i - 1])) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                 << "', the group_list must be an incrementing sequence, but got " << group_list;
      }
    }
  }
  MS_CHECK_VALUE(group_list.back() == expect_sum,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's last element ", group_list.back(), kEqual,
                                                             expect_sum, primitive));

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
