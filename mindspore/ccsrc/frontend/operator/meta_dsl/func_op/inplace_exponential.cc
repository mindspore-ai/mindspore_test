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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/inplace_exponential.h"
#include "mindapi/base/type_id.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::prim {
BeginFunction(InplaceExponential, input, lambd, seed, offset) {
  auto output = Call(Prim(InplaceUniform), input, Value(0.0), Value(1.0), seed, offset);
  output = Call(Prim(InplaceSubScalar), output, Value(1.0), Value(1.0));
  output = Call(Prim(InplaceMuls), output, Value(-1.0));
  output = Call(Prim(InplaceLog), output);
  auto neg_lambd = Call(Prim(ScalarUsub), lambd);
  output = Call(Prim(InplaceDivs), output, neg_lambd);
  Return(output);
}
EndFunction(InplaceExponential)
}  // namespace mindspore::prim
