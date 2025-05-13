/*
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_EINSUM_EXT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_EINSUM_EXT_H_

#include <memory>
#include <vector>
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
void CheckEinsumExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args);
class EinsumExtMetaImpl : public MetaImpl {
 public:
  EinsumExtMetaImpl() : MetaImpl("EinsumExt") {}
  ~EinsumExtMetaImpl() override = default;
  MS_DECLARE_PARENT(EinsumExtMetaImpl, MetaImpl)
  void GenerateFunction() override;

 private:
  NodePtr FastPermute(const NodePtr &input, const std::vector<int64_t> &perm);
  void AdjustOperands(const std::vector<NodePtr> &operands_list, const std::vector<std::vector<int64_t>> &ops_labels,
                      const std::vector<int64_t> &labels_perm_idx, int64_t ellipsis_max_dimnum, int64_t ellipsis_idx,
                      int64_t align_rank, ShapeArray *operands_shape, std::vector<NodePtr> *adjust_operands,
                      std::vector<int64_t> *dim_counts);
  NodePtr Multiplication(const NodePtr &left_operand, const NodePtr &right_operand,
                         const std::vector<int64_t> &sum_dims, ShapeVector *l_shape, ShapeVector r_shape);
  NodePtr ContractOperands(const std::vector<NodePtr> &adjust_operands, const ShapeArray &operands_shape,
                           std::vector<int64_t> dim_counts, int64_t output_rank, int64_t align_rank);
};

REGISTER_META_IMPL(EinsumExt, CheckEinsumExtInputs);
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_EINSUM_EXT_H_
