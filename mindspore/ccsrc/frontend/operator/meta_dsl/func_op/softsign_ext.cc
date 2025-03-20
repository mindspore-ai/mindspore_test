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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/softsign_ext.h"
#include <unordered_set>
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore::prim {
void CheckSoftsignExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const auto &type_id = input_args[kIndex0]->GetType()->cast_ptr<TensorType>()->element()->type_id();
  const std::unordered_set<TypeId> valid_types = {
    kNumberTypeBool,  kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64, kNumberTypeUInt8,
    kNumberTypeFloat, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeDouble};
  if (valid_types.find(type_id) == valid_types.end()) {
    MS_LOG(EXCEPTION) << "For 'Softsign', the type of 'input' must be Tensor[Bool, Int8, Int16, Int32, Int64, UInt8, "
                         "Float16, Float32, Float64], but got "
                      << input_args[kIndex0]->GetType();
  }
}

BeginFunction(SoftsignExt, input_tensor) {
  auto output_tensor = Call(Prim(Abs), input_tensor);
  output_tensor = Call(Prim(AddScalar), output_tensor, Value(1), Value(1));
  output_tensor = Call(Prim(Div), input_tensor, output_tensor);
  Return(output_tensor);
}
EndFunction(SoftsignExt)
}  // namespace mindspore::prim
