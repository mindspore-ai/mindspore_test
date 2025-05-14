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
#include "infer/ops_func_impl/grouped_matmul_v4_transpose.h"

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
int32_t GroupedMatmulV4TransposeFuncImpl::PrivateCheckValidation(const PrimitivePtr &primitive,
                                                                 const InferInfoPtrList &input_infos,
                                                                 int64_t group_type) const {
  if (group_type == -1) {
    return OP_CHECK_SUCCESS;
  }

  const auto input_num = SizeToLong(input_infos.size());
  auto transpose_a = GetTransposeValue(input_infos, input_num + idxes_.transpose_a_offset);
  auto transpose_b = GetTransposeValue(input_infos, input_num + idxes_.transpose_b_offset);
  if (MS_LIKELY(input_infos[idxes_.weight]->IsSequence())) {
    const auto &weight_tensors = input_infos[idxes_.weight]->GetSequenceElements();
    const auto &weight_type = weight_tensors.front()->GetType();
    if (MS_UNLIKELY(weight_type == kNumberTypeInt8 && (transpose_a || !transpose_b))) {
      MS_EXCEPTION(ValueError)
        << "For internal_op'" << primitive->name()
        << "', When the weight_type is int8, transpose_a should be False, transpose_b should be True, but got "
        << transpose_a << " and " << transpose_b;
    }
  }

  if (MS_UNLIKELY(group_type != 0)) {
    MS_EXCEPTION(ValueError) << "For internal_op'" << primitive->name() << "', group_type should be 0, but got "
                             << group_type;
  }

  const auto &group_list_info = input_infos[group_list_idx_];
  if (MS_UNLIKELY(group_list_info->IsNone() ||
                  (!group_list_info->IsSequence() && group_list_info->GetType() != kNumberTypeInt32))) {
    MS_EXCEPTION(ValueError)
      << "For internal_op'" << primitive->name()
      << "', when group_type is not -1, group_list should be 1-D Tensor with int32 elements, but got "
      << group_list_info->DebugInfo();
  }

  return OP_CHECK_SUCCESS;
}

bool GroupedMatmulV4TransposeFuncImpl::GetTransposeValue(const InferInfoPtrList &input_infos,
                                                         int64_t transpose_index) const {
  auto transpose_opt = input_infos[transpose_index]->GetScalarValue<bool>();
  MS_ASSERT(transpose_opt.has_value());
  return transpose_opt.value();
}

}  // namespace ops
}  // namespace mindspore
