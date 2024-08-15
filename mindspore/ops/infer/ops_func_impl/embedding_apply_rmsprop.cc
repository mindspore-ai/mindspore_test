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

#include "infer/ops_func_impl/embedding_apply_rmsprop.h"

#include <map>
#include <string>
#include <set>

#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"


namespace mindspore {
namespace ops {
void EmbeddingApplyRmspropFuncImpl::CheckInputShapes(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  CheckTensorScalarRank(primitive, input_args[kIndex0], "var_handle");
  CheckTensorScalarRank(primitive, input_args[kIndex1], "lr");
  CheckTensorScalarRank(primitive, input_args[kIndex2], "rho");
  CheckTensorScalarRank(primitive, input_args[kIndex3], "momentum");
  CheckTensorScalarRank(primitive, input_args[kIndex4], "epsilon");
  CheckTensorScalarRank(primitive, input_args[kIndex7], "global_step");
}

void EmbeddingApplyRmspropFuncImpl::CheckInputTypes(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  auto var_handle_type = input_args[kIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var_handle", var_handle_type, {kInt32}, prim_name);

  auto keys_type = input_args[kIndex6]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  const std::set<TypePtr> grad_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> type_dict;
  auto grad_type = input_args[kIndex5]->GetType();
  type_dict.emplace("grad", grad_type);
  auto lr_type = input_args[kIndex1]->GetType();
  type_dict.emplace("lr", lr_type);
  auto rho_type = input_args[kIndex2]->GetType();
  type_dict.emplace("rho", rho_type);
  auto momentum_type = input_args[kIndex3]->GetType();
  type_dict.emplace("momentum", momentum_type);
  auto epsilon_type = input_args[kIndex4]->GetType();
  type_dict.emplace("epsilon", epsilon_type);
  CheckAndConvertUtils::CheckTensorTypeSame(type_dict, grad_types, prim_name);
}
}  // namespace ops
}  // namespace mindspore
