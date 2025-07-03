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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API RepeatInterleaveGrad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  RepeatInterleaveGrad(const SymbolPtr &dout, const SymbolPtr &repeat, const SymbolPtr &dim)
      : InferShapeOp({dout, repeat, dim}) {}
  ~RepeatInterleaveGrad() override = default;
  MS_DECLARE_PARENT(RepeatInterleaveGrad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr RepeatInterleaveGrad::Eval() {
  auto dout = input_as_sptr<ListSymbol>(kIndex0);
  auto repeats = input_as_sptr<ListSymbol>(kIndex1);
  auto dim = input_as<IntSymbol>(kIndex2);
  if (!dout->HasData() || !repeats->HasData() || !dim->HasData()) {
    return GenVList();
  }

  DoNotEvalOnRun();
  auto axis = LongToSize(NormAxis(dim->value(), dout->size()));
  auto result = dout->symbols();
  if (repeats->size() == 1) {
    result[axis] = Emit(std::make_shared<ScalarDiv>(result[axis], repeats->item(0)));
  } else {
    result[axis] = GenInt(repeats->size());
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("RepeatInterleaveGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) {
    auto dout = b->GetInputShape(kIndex0);
    auto repeat = b->GetInputValue(kIndex1);
    if (repeat->is<IntSymbol>()) {
      repeat = ListSymbol::Make({repeat});
    }
    auto dim = b->GetInputValue(kIndex2);
    return b->Emit(std::make_shared<RepeatInterleaveGrad>(dout, ListSymbol::Make({repeat}), dim));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
