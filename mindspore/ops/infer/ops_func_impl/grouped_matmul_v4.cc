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

#include "include/common/utils/utils.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"

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
    if (i == idxes_.group_list) {
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

int32_t GroupedMatmulV4FuncImpl::PrivateCheckValidation(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos, int64_t group_type) const {
  if (group_type == -1) {
    return OP_CHECK_SUCCESS;
  }
  // check the value of group_list
  ShapeVector x_shape;
  if (input_infos[idxes_.x]->IsSequence()) {
    x_shape = input_infos[idxes_.x]->GetSequenceElements().at(kIndex0)->GetShape();
  } else {
    x_shape = input_infos[idxes_.x]->GetShape();
  }
  auto group_list_type_opt = input_infos[group_list_type_idx_]->GetScalarValue<int64_t>();
  auto group_list_opt = input_infos[idxes_.group_list]->GetArrayValue<int64_t>();
  if (MS_UNLIKELY(!group_list_type_opt.has_value() || !group_list_opt.has_value() || IsDynamic(x_shape))) {
    return OP_CHECK_RETRY;
  }

  auto expect_sum = group_type == 0 ? x_shape.front() : x_shape.back();
  const auto &group_list = group_list_opt.value().ToVector();
  auto group_list_type = group_list_type_opt.value();
  if (MS_UNLIKELY(group_list_type != 0 && group_list_type != 1)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_list_type should be 0 or 1, but got "
                             << group_list_type;
  }
  if (group_list_type == 0) {
    for (size_t i = 0; i < group_list.size(); ++i) {
      if (i == kIndex0) {
        MS_CHECK_VALUE(group_list[i] >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                             "element of group_list", group_list[i], kGreaterEqual, 0, primitive));
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
  } else {
    for (auto &e : group_list) {
      MS_CHECK_VALUE(
        e >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg("element of group_list", e, kGreaterEqual, 0, primitive));
    }
    auto actual_sum = std::accumulate(group_list.begin(), group_list.end(), 0);
    MS_CHECK_VALUE(actual_sum == expect_sum, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "sum of group_list", actual_sum, kEqual, expect_sum, primitive));
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
