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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_TRANSPOSE_BATCH_MATMUL_TRANSPOSE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_TRANSPOSE_BATCH_MATMUL_TRANSPOSE_FUSION_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/ops/op_def/math_ops.h"

namespace mindspore {
namespace opt {
class TransposeBatchMatmulTranspose : public PatternProcessPass {
 public:
  explicit TransposeBatchMatmulTranspose(bool multigraph = true,
                                         const string &pass_name = "transpose_batch_matmul_transpose_fusion")
      : PatternProcessPass(pass_name, multigraph) {}
  ~TransposeBatchMatmulTranspose() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  ShapeVector GetPermValue(const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_TRANSPOSE_BATCH_MATMUL_TRANSPOSE_FUSION_H_
