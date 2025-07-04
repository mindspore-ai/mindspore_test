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

#ifndef MINDSPORE_OPS_INFER_ALL_TO_ALL_ALL_GATHER_BATCHMATMUL_H_
#define MINDSPORE_OPS_INFER_ALL_TO_ALL_ALL_GATHER_BATCHMATMUL_H_

#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAlltoAllAllGatherBatchMatMul = "AlltoAllAllGatherBatchMatMul";
/// \brief Fused ops for AlltoAll & AllGather & BatchMatmul
/// Refer to Python API @ref mindspore.ops
class OPS_API AlltoAllAllGatherBatchMatMul : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AlltoAllAllGatherBatchMatMul);
  /// \brief Constructor.
  AlltoAllAllGatherBatchMatMul() : BaseOperator(kNameAlltoAllAllGatherBatchMatMul) {
    InitIOName({"x", "weight", "bias"}, {"y1", "y2", "y3"});
  }
};
AbstractBasePtr AlltoAllAllGatherBatchMatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args);
using AlltoAllAllGatherBatchMatMulPtr = std::shared_ptr<AlltoAllAllGatherBatchMatMul>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ALL_TO_ALL_ALL_GATHER_BATCHMATMUL_H_
