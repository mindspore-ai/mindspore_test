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

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("Meshgrid")
  .SetShapeDependN<DependOn::kShape>()
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto inputs = b->GetSymbolsOfDepend();
    if (inputs.size() == 1 && inputs[0]->is<ListSymbol>()) {
      inputs = inputs[0]->as<ListSymbol>()->symbols();
    }
    if (inputs.size() <= 1) {
      MS_EXCEPTION(ValueError) << "For 'Meshgrid', the number of input tensors should be greater than 1, but got "
                               << inputs.size();
    }
    auto indexing = b->GetAttr("indexing");
    MS_EXCEPTION_IF_NULL(indexing);
    SymbolPtrList out;
    out.reserve(inputs.size());
    for (auto &inp : inputs) {
      if (!inp->HasData()) {
        return nullptr;
      }
      out.emplace_back(inp->as<ListSymbol>()->item(0));
    }
    if (indexing->as<StrSymbol>()->value() == "xy") {
      std::swap(out[0], out[1]);
    }
    // when the input num is N, the outputs are N same shape.
    return ListSymbol::Make(SymbolPtrList(inputs.size(), ListSymbol::Make(std::move(out))));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
