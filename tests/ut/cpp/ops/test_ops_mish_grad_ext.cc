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
#include "infer/ops_func_impl/mish_grad_ext.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_INFER_TEST_DECLARE(MishGradExt, MultiInputOpParams);
OP_FUNC_IMPL_INFER_TEST_CASES(
  MishGradExt,
  testing::Values(MultiInputOpParams{{{1, 3}, {1, 3}}, {kFloat32, kFloat32}, {{1, 3}}, {kFloat32}, {}},
                  MultiInputOpParams{{{1, 3}, {1, 3}}, {kFloat16, kFloat16}, {{1, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-1, 3}, {-1, 3}}, {kFloat16, kFloat16}, {{-1, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-1, 2, 3}, {-1, 2, 3}}, {kFloat32, kFloat32}, {{-1, 2, 3}}, {kFloat32}, {}},
                  MultiInputOpParams{{{-2}, {-2}}, {kFloat32, kFloat32}, {{-2}}, {kFloat32}, {}}));
}  // namespace ops
}  // namespace mindspore
