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

#include "infer/ops_func_impl/rand_ext.h"
#include <set>
#include <vector>
#include <memory>
#include <string>
#include "infer/ops_func_impl/ones.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "op_def/op_name.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
void CheckRandIntRange(const InferInfoPtr &from, const InferInfoPtr &to, const std::string &name, bool output_bool) {
  auto from_opt = from->GetScalarValue<int64_t>();
  auto to_opt = to->GetScalarValue<int64_t>();
  if (!from_opt.has_value() || !to_opt.has_value()) {
    return;
  }
  auto from_value = from_opt.value();
  auto to_value = to_opt.value();
  MS_ASSERT_TRUE(from_value < to_value) << "For " << name << ", expected 'from' is less than 'to', but got "
                                        << from_value << " and " << to_value;
  if (output_bool) {
    CheckAndConvertUtils::CheckInRange("from", from_value, kIncludeBoth, {0, 1}, name);
    CheckAndConvertUtils::CheckInRange("to", from_value, kIncludeBoth, {0, 1}, name);
  }
}

TypeIdList RandExtFuncImpl ::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  auto &dtype = input_infos[dtype_idx_];
  if (dtype->IsNone()) {
    return prim_name == kNameRandInt ? TypeIdList{kNumberTypeInt64} : TypeIdList{kNumberTypeFloat32};
  }
  auto infer_type = static_cast<TypeId>(dtype->GetScalarValueWithCheck<int64_t>());
  if (prim_name == kNameRandInt) {
    CheckRandIntRange(input_infos[kInputIndex0], input_infos[kInputIndex1], prim_name, (infer_type == kNumberTypeBool));
  }
  CheckAndConvertUtils::CheckTypeIdValid("dtype", infer_type, common_mint_valid_type_ids_with_bool, primitive->name());
  return {infer_type};
}
}  // namespace ops
}  // namespace mindspore
