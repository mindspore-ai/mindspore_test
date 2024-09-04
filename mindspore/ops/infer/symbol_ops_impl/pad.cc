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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_add.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Pad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Pad(const SymbolPtr &input, const SymbolPtr &padding) : InferShapeOp({input, padding}) {}
  ~Pad() override = default;
  MS_DECLARE_PARENT(Pad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Pad::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  auto paddings = input_as<ListSymbol>(kIndex1);
  if (!input->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto result = input->symbols();
  for (size_t i = 0; i < input->size(); i++) {
    auto padding = paddings->item_as<ListSymbol>(i);
    auto p = Emit(std::make_shared<ScalarAdd>(padding->item(0), padding->item(1)));
    result[i] = Emit(std::make_shared<ScalarAdd>(result[i], p));
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Pad").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto input = b->GetInputShape(kIndex0);
  auto paddings = b->GetAttr("paddings");
  MS_EXCEPTION_IF_NULL(paddings);
  return b->Emit(std::make_shared<Pad>(input, paddings));
});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
