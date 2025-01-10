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
#include <memory>
#include "common/common_test.h"
#include "abstract/dshape.h"
#include "infer/ops_func_impl/moe_init_routing.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(MoeInitRouting, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(MoeInitRouting, testing::Values(MultiInputOpParams{{{10, 200}, {10, 2}, {10, 2}},
                                                                           {kFloat16, kInt32, kInt32},
                                                                           {{20, 200}, {20}, {20}},
                                                                           {kFloat16, kInt32, kInt32},
                                                                           {CreateScalar<int64_t>(10)}}));
}  // namespace ops
}  // namespace mindspore
