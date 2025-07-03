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

#include "infer/ops_func_impl/exp2.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
ShapeArray Exp2FuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> Exp2FuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  TypeId input_type = input_infos[kIndex0]->GetType();
  // The supported ACLNN types are BOOL, INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT32, and BFLOAT16.
  // In addition, all input types except FLOAT16 / BFLOAT16 need to be converted to FLOAT32 output.
  static const std::vector<TypeId> need_covert_types = {kNumberTypeBool,  kNumberTypeInt8,  kNumberTypeInt16,
                                                        kNumberTypeInt32, kNumberTypeInt64, kNumberTypeUInt8};
  bool need_covert = std::any_of(need_covert_types.begin(), need_covert_types.end(),
                                 [&input_type](const TypeId &type) { return input_type == type; });
  if (need_covert) {
    input_type = kNumberTypeFloat32;
  }
  return {input_type};
}
}  // namespace mindspore::ops
