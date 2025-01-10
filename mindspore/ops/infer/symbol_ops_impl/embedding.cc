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
class OPS_API Embedding : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Embedding(const SymbolPtr &inp, const SymbolPtr &axis) : InferShapeOp({inp, axis}) {}
  ~Embedding() override = default;
  MS_DECLARE_PARENT(Embedding, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Embedding::Eval() {
  auto input_shape = input_as<ListSymbol>(kIndex0);
  auto weight_shape = input_as<ListSymbol>(kIndex1);
  if (!input_shape->HasData()) {
    return GenVList();
  }
  auto result = input_shape->symbols();
  if (weight_shape->HasData()) {
    // weight is 2-D tensor.
    (void)result.emplace_back(weight_shape->item(kIndex1));
  } else {
    (void)result.emplace_back(GenVInt());
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Embedding").SetShapeDepend({DependOn::kShape, DependOn::kShape}).SetShapeFuncWith<Embedding>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
