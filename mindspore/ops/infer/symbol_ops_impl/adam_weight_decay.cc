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
#include "symbolic_shape/operation_builder.h"
#include "infer/symbol_ops_impl/common.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr AdamWeigtDecayShapeBuilder(OperationBuilder *b) {
  // var, m, v
  return ListSymbol::Make({b->GetInputShape(kIndex0), b->GetInputShape(kIndex1), b->GetInputShape(kIndex2)});
}

REG_SYMBOL_OP_BUILDER("AdamWeightDecay")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(AdamWeigtDecayShapeBuilder);
REG_SYMBOL_OP_BUILDER("AdamW")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(AdamWeigtDecayShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
