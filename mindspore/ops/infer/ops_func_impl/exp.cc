/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/exp.h"
#include <memory>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
std::vector<TypeId> ExpFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input_type_id = input_infos[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kInt64,   kBool,      kFloat16,    kFloat32,
                                         kFloat64, kComplex64, kComplex128, kBFloat16};
  const auto &op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTypeValid("input", TypeIdToType(input_type_id), valid_types, op_name);

  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                                  kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                                  kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_int_or_bool) {
    return {kNumberTypeFloat32};
  }
  return {input_type_id};
}
}  // namespace ops
}  // namespace mindspore
