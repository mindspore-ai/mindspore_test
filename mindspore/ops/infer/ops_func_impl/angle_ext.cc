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

#include "infer/ops_func_impl/angle_ext.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {
ShapeArray AngleExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> AngleExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  TypeId type_id;
  TypeId input_type = input_infos[kIndex0]->GetType();
  static const std::vector<TypeId> supported_dtypes = {kNumberTypeBool,    kNumberTypeInt8,    kNumberTypeInt16,
                                                       kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,
                                                       kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeComplex64};
  bool is_supported = std::any_of(supported_dtypes.begin(), supported_dtypes.end(),
                                  [&input_type](const TypeId &type) { return input_type == type; });
  if (!is_supported) {
    MS_EXCEPTION(TypeError) << "For `AngleExt`, the input tensor dtype " << TypeIdToString(input_type, true)
                            << " is not supported.";
  }
  if (input_type == kNumberTypeFloat16) {
    type_id = kNumberTypeFloat16;
  } else {
    type_id = kNumberTypeFloat32;
  }

  return {type_id};
}
}  // namespace ops
}  // namespace mindspore
