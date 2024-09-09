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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("TopKRouter")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x_shape = b->GetInputShape(kIndex0)->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_NULL(x_shape);
    if (!x_shape->HasData()) {
      return nullptr;
    }
    auto capacity_value = b->GetInputValue(kIndex1);
    auto expert_value = b->GetInputValue(kIndex2);
    auto out = ListSymbol::Make({x_shape->item(0), expert_value, capacity_value});
    return ListSymbol::Make({out, x_shape});
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
