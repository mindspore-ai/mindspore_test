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
#include "ops_utils/op_constants.h"
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr CalMask(const SymbolPtrList &symbols, const OperationEmitter &e) {
  auto count = Accumulate<ScalarMul>(symbols, e);
  constexpr int64_t kDropoutGenMaskMaskConvertLen = 128;
  constexpr int64_t kUint8OfDropoutGenMaskMaskConvertLen = 16;
  auto n128s = e.Emit(std::make_shared<ScalarCeilDiv>(count, IntSymbol::Make(kDropoutGenMaskMaskConvertLen)));
  return e.Emit(std::make_shared<ScalarMul>(n128s, IntSymbol::Make(kUint8OfDropoutGenMaskMaskConvertLen)));
}

class OPS_API DropoutExt : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  explicit DropoutExt(const SymbolPtr &x) : InferShapeOp({x}) {}
  ~DropoutExt() override = default;
  MS_DECLARE_PARENT(DropoutExt, InferShapeOp)

 protected:
  SymbolPtr Eval() override {
    auto x = input_as<ListSymbol>(kIndex0);
    if (!x->HasData()) {
      return GenList({input(0), GenList({GenVInt()})});
    }
    return GenList({input(0), GenList({CalMask(x->symbols(), emitter())})});
  }
};

class OPS_API DropoutGenMaskExt : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  explicit DropoutGenMaskExt(const SymbolPtr &x) : InferShapeOp({x}) {}
  ~DropoutGenMaskExt() override = default;
  MS_DECLARE_PARENT(DropoutGenMaskExt, InferShapeOp)

 protected:
  SymbolPtr Eval() override {
    auto x = input_as<ListSymbol>(kIndex0);
    if (!x->HasData()) {
      return GenList({GenVInt()});
    }
    return GenList({CalMask(x->symbols(), emitter())});
  }
};

REG_SYMBOL_OP_BUILDER("Dropout").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto s = b->GetInputShape(kIndex0);
  return ListSymbol::Make({s, s});
});
REG_SYMBOL_OP_BUILDER("DropoutExt").SetShapeDepend({DependOn::kShape}).SetShapeFuncWith<DropoutExt>();
REG_SYMBOL_OP_BUILDER("DropoutDoMaskExt").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("DropoutGenMaskExt").SetShapeDepend({DependOn::kValue}).SetShapeFuncWith<DropoutGenMaskExt>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
