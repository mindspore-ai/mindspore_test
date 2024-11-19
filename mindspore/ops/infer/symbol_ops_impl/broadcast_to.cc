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
class OPS_API BroadcastTo : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  BroadcastTo(const SymbolPtr &x, const SymbolPtr &out_shape) : InferShapeOp({x, out_shape}) {}
  ~BroadcastTo() override = default;
  MS_DECLARE_PARENT(BroadcastTo, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr BroadcastTo::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto out_shape = input_as<ListSymbol>(kIndex1);
  if (!out_shape->HasData()) {
    return GenVList();
  }
  if (x->size() > out_shape->size()) {
    MS_EXCEPTION(ValueError)
      << "For BroadcastTo, the input out_shape's size should be less equal to output size, but got " << x->size()
      << " vs " << out_shape->size();
  }
  auto result = out_shape->symbols();
  for (size_t i = x->size(); i > 0; i--) {
    auto xi = x->item_as_sptr<IntSymbol>(x->size() - i);
    auto res_idx = result.size() - i;
    auto ri = result[res_idx]->as_sptr<IntSymbol>();
    // when out_shape[i] is "-1", directly use the input shape item
    if (ri->is_negative()) {
      result[res_idx] = xi;
    } else if (!ri->is_positive()) {
      if (xi->is_greater_than(1)) {
        result[res_idx] = xi;
      } else {
        result[res_idx] = GenVInt();
      }
    }
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("BroadcastTo")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<BroadcastTo>();

// dynamic_broadcast_to's out shape has not "-1"
REG_SYMBOL_OP_BUILDER("DynamicBroadcastTo")
  .SetShapeDepend({DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
