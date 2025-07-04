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

#include "infer/ops_func_impl/embedding_apply_adam_w.h"

#include <set>
#include <vector>
#include <map>
#include <string>

#include "utils/check_convert_utils.h"
#include "op_def/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
void EmbeddingApplyAdamWFuncImpl::CheckInputShapes(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  CheckTensorScalarRank(primitive, input_args[kInputIndex0], "var_handle");
  CheckTensorScalarRank(primitive, input_args[kInputIndex1], "beta1_power");
  CheckTensorScalarRank(primitive, input_args[kInputIndex2], "beta2_power");
  CheckTensorScalarRank(primitive, input_args[kInputIndex3], "lr");
  CheckTensorScalarRank(primitive, input_args[kInputIndex4], "weight_decay");
  CheckTensorScalarRank(primitive, input_args[kInputIndex5], "beta1");
  CheckTensorScalarRank(primitive, input_args[kInputIndex6], "beta2");
  CheckTensorScalarRank(primitive, input_args[kInputIndex7], "epsilon");
  CheckTensorScalarRank(primitive, input_args[kInputIndex11], "global_step");
}

void EmbeddingApplyAdamWFuncImpl::CheckInputTypes(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto var_handle_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var_handle", var_handle_type, {kInt32}, prim_name);

  auto keys_type = input_args[kInputIndex9]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  const std::set<TypePtr> grad_types = {kFloat16, kFloat32};
  auto grad_type = input_args[kInputIndex8]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, grad_types, prim_name);

  std::map<std::string, TypePtr> type_dict;
  type_dict.emplace("grad", grad_type);

  auto beta1_power_type = input_args[kInputIndex1]->GetType();
  auto beta2_power_type = input_args[kInputIndex2]->GetType();
  auto lr_type = input_args[kInputIndex3]->GetType();
  auto weight_decay_type = input_args[kInputIndex4]->GetType();
  auto beta1_type = input_args[kInputIndex5]->GetType();
  auto beta2_type = input_args[kInputIndex6]->GetType();
  auto epsilon_type = input_args[kInputIndex7]->GetType();
  type_dict.emplace("beta1_power", beta1_power_type);
  type_dict.emplace("beta2_power", beta2_power_type);
  type_dict.emplace("lr", lr_type);
  type_dict.emplace("weight_decay", weight_decay_type);
  type_dict.emplace("beta1", beta1_type);
  type_dict.emplace("beta2", beta2_type);
  type_dict.emplace("epsilon", epsilon_type);

  CheckAndConvertUtils::CheckTensorTypeSame(type_dict, grad_types, prim_name);
}
}  // namespace ops
}  // namespace mindspore
