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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_cast.h"
#include <memory>
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ScalarCastBuilder(OperationBuilder *b) {
  auto s = b->GetInputValue(kIndex0);
  auto tid = AsInt(b->GetInputValue(kIndex1));
  switch (tid) {
    case kNumberTypeInt:
    case kNumberTypeInt8:
    case kNumberTypeInt16:
    case kNumberTypeInt32:
    case kNumberTypeInt64:
    case kNumberTypeUInt:
    case kNumberTypeUInt8:
    case kNumberTypeUInt16:
    case kNumberTypeUInt32:
    case kNumberTypeUInt64:
      return b->Emit(std::make_shared<ScalarCast<IntSymbol>>(s));
    case kNumberTypeFloat:
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeFloat64:
      return b->Emit(std::make_shared<ScalarCast<FloatSymbol>>(s));
    case kNumberTypeBool:
      return b->Emit(std::make_shared<ScalarCast<BoolSymbol>>(s));
    default:
      return nullptr;
  }
}

REG_SYMBOL_OP_BUILDER("ScalarCast").SetValueDependN<DependOn::kValue, 2>().SetValueFunc(ScalarCastBuilder);
REG_SYMBOL_OP_BUILDER("ScalarToTensor").SetValueDependN<DependOn::kValue, 2>().SetValueFunc(ScalarCastBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
