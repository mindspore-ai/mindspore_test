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
class OPS_API MoeTokenPermuteGrad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~MoeTokenPermuteGrad() override = default;
  MS_DECLARE_PARENT(MoeTokenPermuteGrad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr MoeTokenPermuteGrad::Eval() {
  constexpr size_t out_permuted_tokens_shape_rank = 2;
  auto grad = input_as<ListSymbol>(kIndex0);
  auto sorted_indices_shape = input_as<ListSymbol>(kIndex1);
  auto num_topk = input(kIndex2);
  SymbolPtrList out_permuted_tokens_shape(out_permuted_tokens_shape_rank);
  if (grad->HasData()) {
    out_permuted_tokens_shape[kIndex1] = grad->item(kIndex1);
  }
  if (!sorted_indices_shape->HasData()) {
    out_permuted_tokens_shape[kIndex0] = GenVInt();
    return GenList(std::move(out_permuted_tokens_shape));
  }
  DoNotEvalOnRun();
  out_permuted_tokens_shape[kIndex0] = Emit(std::make_shared<ScalarDiv>(sorted_indices_shape->item(0), num_topk));
  return ResultIntList(std::move(out_permuted_tokens_shape));
}

REG_SYMBOL_OP_BUILDER("MoeTokenPermuteGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kNone})
  .SetShapeFuncWith<MoeTokenPermuteGrad>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
