/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
class OPS_API OneHot : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  OneHot(const SymbolPtr &indices, const SymbolPtr &depth, const SymbolPtr &axis)
      : InferShapeOp({indices, depth, axis}) {}
  ~OneHot() override = default;
  MS_DECLARE_PARENT(OneHot, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr OneHot::Eval() {
  auto indices = input_as<ListSymbol>(kIndex0);
  auto depth = input(kIndex1);
  if (!indices->HasData()) {
    return GenVList();
  }
  auto axis_sym = input_as<IntSymbol>(kIndex2);
  if (!axis_sym->HasData()) {
    return GenVIntList(indices->size() + 1);
  }
  DoNotEvalOnRun();
  int64_t axis = axis_sym->value();
  SymbolPtrList result = indices->symbols();
  if (axis >= 0) {
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(axis) <= result.size(), "axis out of range of input size");
    (void)result.insert(result.begin() + static_cast<size_t>(axis), depth);
  } else {
    (void)result.emplace_back(depth);
  }
  return ResultIntList(std::move(result));
}

SymbolPtr OneHotShapeBuilder(OperationBuilder *b) {
  auto indices = b->GetInputShape(kIndex0);
  auto depth = b->GetInputValue(kIndex1);
  auto axis = b->GetInputOrAttr(kIndex4, kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis);
  return b->Emit(std::make_shared<OneHot>(indices, depth, axis));
}

REG_SYMBOL_OP_BUILDER("OneHot")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(OneHotShapeBuilder);
REG_SYMBOL_OP_BUILDER("OneHotExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(OneHotShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
