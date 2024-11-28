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
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "infer/ops_func_impl/isneginf.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore {
namespace ops {
namespace {

std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator.FeedInputArgs({InferInfoParam{ShapeVector{4, 2, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{4, 2, 3}}, {kNumberTypeBool});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{4, 2, 3, 6}, kNumberTypeFloat16}})
    .FeedExpectedOutput({{4, 2, 3, 6}}, {kNumberTypeBool});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeInt32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeBool});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{4, -1, 3}, kNumberTypeBFloat16}})
    .FeedExpectedOutput({{4, -1, 3}}, {kNumberTypeBool});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(IsNegInf, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace ops
}  // namespace mindspore