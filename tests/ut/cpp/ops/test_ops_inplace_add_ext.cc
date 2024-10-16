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
#include "infer/ops_func_impl/inplace_add_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_INFER_TEST_DECLARE(InplaceAddExt, MultiInputOpParams);

OP_FUNC_IMPL_INFER_TEST_CASES(
  InplaceAddExt,
  testing::Values(MultiInputOpParams{{{1}, {1}}, {kFloat16, kFloat16}, {{1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kFloat16, kFloat32}, {{2, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
                  MultiInputOpParams{{{2, 3, 4}, {2, 3, 4}}, {kFloat64, kFloat64}, {{2, 3, 4}}, {kFloat64}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kInt8, kFloat32}, {{2, 3}}, {kInt8}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kInt16, kFloat32}, {{2, 3}}, {kInt16}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kInt32, kFloat32}, {{2, 3}}, {kInt32}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kInt64, kFloat32}, {{2, 3}}, {kInt64}, {}},
                  MultiInputOpParams{{{2, 3}, {3}}, {kUInt8, kUInt8}, {{2, 3}}, {kUInt8}, {}},
                  MultiInputOpParams{{{-1, -1}, {1, 2}}, {kBool, kBool}, {{-1, -1}}, {kBool}, {}}));
}  // namespace ops
}  // namespace mindspore
