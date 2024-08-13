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
#include "mindspore/ops/infer/symbol_ops_impl/addn.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("Select")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape})
  .SetShapeFunc([](OperationBuilder *b) {
    return AddnBuildShape(b, {b->GetInputShape(kIndex0), b->GetInputShape(kIndex1), b->GetInputShape(kIndex2)});
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
