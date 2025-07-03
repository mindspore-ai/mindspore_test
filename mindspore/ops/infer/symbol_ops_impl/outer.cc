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

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Outer : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Outer() override = default;
  MS_DECLARE_PARENT(Outer, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Outer::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto y = input_as<ListSymbol>(kIndex1);
  if (x->HasData() && y->HasData()) {
    DoNotEvalOnRun();
  }
  SymbolPtr ret0 = x->HasData() ? x->item(kIndex0) : GenVInt();
  SymbolPtr ret1 = y->HasData() ? y->item(kIndex0) : GenVInt();
  return GenList({ret0, ret1});
}

REG_SYMBOL_OP_BUILDER("Outer").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<Outer>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
