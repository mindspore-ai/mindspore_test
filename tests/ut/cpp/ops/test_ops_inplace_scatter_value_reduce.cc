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

#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
template <typename s>
std::vector<InferInfoParam> scatter_val_input(ShapeVector shape, TypeId dtype, int64_t dim) {
  return {
    InferInfoParam{ShapeVector{shape}, dtype},                                  // input
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(dim)},         // dim
    InferInfoParam{ShapeVector{shape}, kNumberTypeInt64},                       // index
    InferInfoParam{ShapeVector{}, dtype, CreateScalar<s>(0)},                   // value
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // reduce
  };
}

std::vector<GeneralInferParam> prepare_params() {
  // Only shape and type of the first operand matters. Ignore uncecessary dim/input/value/reduce.
  GeneralInferParamGenerator generator;
  return generator.FeedInputArgs(scatter_val_input<double>({1}, kNumberTypeFloat64, 0))
    .FeedExpectedOutput({{1}}, {kNumberTypeFloat64})
    .FeedInputArgs(scatter_val_input<float>({2, 3}, kNumberTypeFloat16, 1))
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16})
    .FeedInputArgs(scatter_val_input<float>({2, 3, 4}, kNumberTypeFloat32, 2))
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeFloat32})
    .FeedInputArgs(scatter_val_input<bool>({-1, -1}, kNumberTypeBool, 2))
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeBool})
    .FeedInputArgs(scatter_val_input<int32_t>({-2}, kNumberTypeComplex128, 2))
    .FeedExpectedOutput({{-2}}, {kNumberTypeComplex128})
    .Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InplaceScatterValueReduce, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
