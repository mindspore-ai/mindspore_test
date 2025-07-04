/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/irpass/row_tensor_eliminate.h"

#include "mindspore/ops/op_def/sparse_tensor_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/pattern_matcher.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimRowTensorGetIndices, {prim::kPrimMakeRowTensor, Xs}}
// {prim::kPrimRowTensorGetValues, {prim::kPrimMakeRowTensor, Xs}}
// {prim::kPrimRowTensorGetDenseShape, {prim::kPrimMakeRowTensor, Xs}}

AnfNodePtr RowTensorEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x;
  PatternNode y;
  PatternNode z;
  auto slices = PPrimitive(prim::kPrimMakeRowTensor, x, y, z).MinExtraNodes(0);
  MATCH_REPLACE(node, PPrimitive(prim::kPrimRowTensorGetIndices, slices), x);
  MATCH_REPLACE(node, PPrimitive(prim::kPrimRowTensorGetValues, slices), y);
  MATCH_REPLACE(node, PPrimitive(prim::kPrimRowTensorGetDenseShape, slices), z);
  return nullptr;
}

// {prim::kPrimRowTensorAdd, rowtensor, zeros_like(x)} -> rowtensor

AnfNodePtr RowTensorAddZerosLike::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x;
  PatternNode y;
  auto zeros_like = PPrimitive(prim::kPrimZerosLike, y);
  MATCH_REPLACE(node, PPrimitive(prim::kPrimRowTensorAdd, x, zeros_like), x);
  return nullptr;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
