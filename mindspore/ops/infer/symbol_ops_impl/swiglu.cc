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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Swiglu : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Swiglu(const SymbolPtr &inp, const SymbolPtr &axis) : InferShapeOp({inp, axis}) {}
  ~Swiglu() override = default;
  MS_DECLARE_PARENT(Swiglu, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Swiglu::Eval() {
  auto x_shape = input_as<ListSymbol>(kIndex0);
  auto axis = input_as<IntSymbol>(kIndex1);
  if (!x_shape->HasData()) {
    return GenVList();
  }
  if (!axis->HasData()) {
    return GenVIntList(x_shape->size());
  }
  auto dim = LongToSize(NormAxis(axis->value(), x_shape->size()));
  auto result = x_shape->symbols();
  result[dim] = Emit(std::make_shared<ScalarDiv>(result[dim], kSym2));
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Swiglu").SetShapeDepend({DependOn::kShape, DependOn::kValue}).SetShapeFuncWith<Swiglu>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
