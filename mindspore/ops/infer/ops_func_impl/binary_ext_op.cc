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

#include "infer/ops_func_impl/binary_ext_op.h"
#include <vector>
#include <map>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
static inline bool IsIntegralBinaryType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}
ShapeArray BinaryExtOpFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto output_shape = input_infos[kInputIndex0]->GetShape();
  std::vector<std::string> input_names = {"input", "other", "alpha"};
  for (size_t i = 1; i < input_infos.size(); ++i) {
    auto input_shape = input_infos[i]->GetShape();
    output_shape = CalBroadCastShape(output_shape, input_shape, primitive->name(), input_names[i - 1], input_names[i]);
  }
  return {output_shape};
}

std::vector<TypeId> BinaryExtOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto input = input_infos[kInputIndex0]->GetType();
  auto other = input_infos[kInputIndex1]->GetType();
  auto alpha = input_infos[kInputIndex2]->GetType();
  auto typePtr = TypeIdToType(input);
  if (alpha == kNumberTypeFloat32 && (IsIntegralBinaryType(input) || IsIntegralBinaryType(other))) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', floating alpha need floating input and other, but got " << TypeIdToString(input)
                            << " and " << TypeIdToString(other);
  }
  if (alpha == kNumberTypeBool && (input != kNumberTypeBool || other != kNumberTypeBool)) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', boolean alpha need boolean input and other, but got "
                            << TypeIdToString(input) << " and " << TypeIdToString(other);
  }
  if (input != other) {
    MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                            << "], the input arguments must have same data type, but got Tensor["
                            << TypeIdToString(input) << "] and Tensor[" << TypeIdToString(other) << "]";
  }
  if (!std::any_of(common_valid_types_with_complex_and_bool.begin(), common_valid_types_with_complex_and_bool.end(),
                   [&typePtr](TypePtr accept) { return typePtr == accept; })) {
    MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                            << "], the input arguments must have error data type, got Tensor[" << TypeIdToString(input)
                            << "] and Tensor[" << TypeIdToString(other) << "]";
  }
  return {input};
}

}  // namespace ops
}  // namespace mindspore
