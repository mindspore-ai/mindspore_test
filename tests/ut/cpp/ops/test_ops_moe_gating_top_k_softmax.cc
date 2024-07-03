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
#include "common/common_test.h"
#include "abstract/dshape.h"
#include "infer/ops_func_impl/moe_gating_top_k_softmax.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
auto dyn_rank = abstract::TensorShape::kShapeRankAny;
auto dyn_dim = abstract::TensorShape::kShapeDimAny;

OP_FUNC_IMPL_TEST_DECLARE(MoeGatingTopKSoftmax, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  MoeGatingTopKSoftmax,
  testing::Values(
    MultiInputOpParams{{{10, 200}, {10}}, {kFloat16, kBool},
                       {{10, 2}, {10, 2}, {10, 2}}, {kFloat16, kInt32, kInt32},
                       {CreateScalar<int64_t>(2)}},
    MultiInputOpParams{{{10, 200}, {10}}, {kFloat16, kBool},
                       {{10, dyn_dim}, {10, dyn_dim}, {10, dyn_dim}}, {kFloat16, kInt32, kInt32},
                       {CreateScalar(kValueAny)}},
    MultiInputOpParams{{{dyn_dim, 200}, {10}}, {kFloat16, kBool},
                       {{dyn_dim, 2}, {dyn_dim, 2}, {dyn_dim, 2}}, {kFloat16, kInt32, kInt32},
                       {CreateScalar<int64_t>(2)}},
    MultiInputOpParams{{{dyn_rank}, {10}}, {kFloat16, kBool},
                       {{dyn_rank}, {dyn_rank}, {dyn_rank}}, {kFloat16, kInt32, kInt32},
                       {CreateScalar<int64_t>(2)}}
  ));
}  // namespace ops
}  // namespace mindspore
