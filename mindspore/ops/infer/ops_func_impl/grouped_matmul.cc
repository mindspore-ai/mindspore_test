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

#include "infer/ops_func_impl/grouped_matmul.h"

#include <vector>

#include "include/common/utils/utils.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
/*
separated means the size of tensorlist not equal 1.
integrated means the size of tensorlist is 1.
split_item        inputs     weight      outputs
      0:      separated     separated    separated
      1:     integrated     b, k, n      separated
      2:      separated     separated    integrated
      3:     integrated     b, k, n      integrated
*/
void GroupedMatmulFuncImpl::FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
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

int64_t GroupedMatmulFuncImpl::FetchGroupListSize(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  const auto &group_list_shape = input_infos[idxes_.group_list]->GetShape();
  MS_CHECK_VALUE(group_list_shape.size() == kIndex1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's rank", group_list_shape.size(), kEqual,
                                                             kIndex1, primitive));
  return input_infos[idxes_.group_list]->IsDynamic() ? abstract::Shape::kShapeDimAny : group_list_shape[kIndex0];
}

int32_t GroupedMatmulFuncImpl::PrivateCheckValidation(const PrimitivePtr &primitive,
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

  const auto &group_list_info = input_infos[idxes_.group_list];

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool enable_infer_boost_310p =
    ms_context->IsEnableInferBoost() && ms_context->ascend_soc_version() == kAscendVersion310p;
  if (enable_infer_boost_310p) {
    auto transpose_a = GetTransposeValue(input_infos, idxes_.transpose_a);
    auto transpose_b = GetTransposeValue(input_infos, idxes_.transpose_b);
    if (MS_UNLIKELY(transpose_a || !transpose_b)) {
      MS_EXCEPTION(ValueError) << "For internal_op'" << primitive->name()
                               << "', transpose_a should be False, transpose_b should be True, but got " << transpose_a
                               << " and " << transpose_b;
    }

    if (MS_UNLIKELY(group_type != 0)) {
      MS_EXCEPTION(ValueError) << "For internal_op'" << primitive->name() << "', group_type should be 0, but got "
                               << group_type;
    }

    if (MS_UNLIKELY(group_list_info->IsNone() ||
                    (!group_list_info->IsSequence() && group_list_info->GetType() != kNumberTypeInt32))) {
      MS_EXCEPTION(ValueError) << "For internal_op'" << primitive->name()
                               << "', when group_list should be 1-D Tensor or List with int32 elements, but got "
                               << group_list_info->DebugInfo();
    }
  }

  auto group_list_opt = group_list_info->GetArrayValue<int64_t>();
  if (MS_UNLIKELY(IsDynamic(x_shape) || !group_list_opt.has_value() || group_list_opt.value().HasUnknownValue())) {
    return OP_CHECK_RETRY;
  }

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
  if (!enable_infer_boost_310p) {
    auto expect_sum = group_type == 0 ? x_shape.front() : x_shape.back();
    MS_CHECK_VALUE(group_list.back() == expect_sum,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's last element ", group_list.back(), kEqual,
                                                               expect_sum, primitive));
  }

  return OP_CHECK_SUCCESS;
}

bool GroupedMatmulFuncImpl::GetTransposeValue(const InferInfoPtrList &input_infos, size_t transpose_index) const {
  return static_cast<bool>(input_infos[transpose_index]->GetScalarValue<bool>().value());
}
}  // namespace ops
}  // namespace mindspore
