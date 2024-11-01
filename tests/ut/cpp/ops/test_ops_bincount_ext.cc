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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/bincount_ext.h"

namespace mindspore {
namespace ops {

OP_FUNC_IMPL_INFER_TEST_DECLARE(BincountExt, MultiInputOpParams);

OP_FUNC_IMPL_INFER_TEST_CASES(
  BincountExt,
  testing::Values(
    MultiInputOpParams{{{3}, {3}}, {kInt32, kFloat32}, {{3}}, {kFloat32}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{9}, {9}}, {kInt32, kFloat64}, {{9}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{9}, {9}}, {kInt32, kInt32}, {{9}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{9}, {9}}, {kInt32, kFloat16}, {{9}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{9}, {9}}, {kInt32, kBool}, {{9}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{100}, {100}}, {kInt32, kUInt8}, {{100}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{{{-2}, {-2}}, {kInt32, kInt32}, {{-1}}, {kFloat64}, {CreateScalar<int64_t>(kNumberTypeInt32)}},
    MultiInputOpParams{
      {{-1}, {-1}}, {kInt32, kFloat32}, {{-1}}, {kFloat32}, {CreateScalar<int64_t>(kNumberTypeInt32)}}));

}  // namespace ops
}  // namespace mindspore
