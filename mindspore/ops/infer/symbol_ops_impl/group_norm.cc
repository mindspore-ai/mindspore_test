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
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API GroupNorm : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  GroupNorm(const SymbolPtr &x, const SymbolPtr &num_groups) : InferShapeOp({x, num_groups}) {}
  ~GroupNorm() override = default;
  MS_DECLARE_PARENT(GroupNorm, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr GroupNorm::Eval() {
  auto x = input_as_sptr<ListSymbol>(kIndex0);
  auto num_groups = input(kIndex1);
  if (!x->HasData()) {
    auto out_shape = GenList({GenVInt(), num_groups});
    return GenList({x, out_shape, out_shape});
  }
  DoNotEvalOnRun();
  auto out_shape = GenList({x->item(kIndex0), num_groups});
  return GenList({x, out_shape, out_shape});
}

REG_SYMBOL_OP_BUILDER("GroupNorm").SetShapeDepend({DependOn::kShape, DependOn::kValue}).SetShapeFuncWith<GroupNorm>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
