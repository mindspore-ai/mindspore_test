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

#include "infer/ops_func_impl/nansum.h"
#include <set>
#include <memory>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {

ShapeArray NansumFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return ReduceGeneralInferShape(primitive, input_infos);
}

std::vector<TypeId> NansumFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto &dtype = input_infos[kIndex3];
  TypeId output_type;
  if (dtype->IsNone()) {
    const auto &input_type_id = input_infos[kIndex0]->GetType();
    static std::set<TypeId> intergral_set = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                             kNumberTypeInt32};
    if (intergral_set.find(input_type_id) != intergral_set.end()) {
      output_type = kNumberTypeInt64;
    } else {
      output_type = input_type_id;
    }
  } else {
    output_type = static_cast<TypeId>(dtype->GetScalarValueWithCheck<int64_t>());
  }
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
