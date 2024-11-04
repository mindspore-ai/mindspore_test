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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/gcd.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
    GeneralInferParamGenerator generator;
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt16}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt16}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt16});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt32}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt32}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt32});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}, InferInfoParam{ShapeVector{2, 1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{4, 4, 3, 2}, kNumberTypeInt64}, InferInfoParam{ShapeVector{1, 4, 3, 2}, kNumberTypeInt64}})
      .FeedExpectedOutput({{4, 4, 3, 2}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-1}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{1, 3}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-1, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{1, 3}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-1, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{1, 1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{-1, -1, 1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{1, 1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{-1, -1, 1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{5, 3, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{3, -1, -1, 2}, kNumberTypeInt64}})
      .FeedExpectedOutput({{3, 5, 3, 2}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 1, 1, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{-1, -1, 1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{2, -1, -1, -1}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{4, 1, -1, 9}, kNumberTypeInt64}, InferInfoParam{ShapeVector{3, -1, 4, -1, 5, 9}, kNumberTypeInt64}})
      .FeedExpectedOutput({{3, -1, 4, -1, 5, 9}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{3, -1, 4, -1, 5, 9, 3, 1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{5, -1, -1, -1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{3, -1, 4, -1, 5, 9, 3, -1}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-2}, kNumberTypeInt64}, InferInfoParam{ShapeVector{3, -1}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{3, -1}, kNumberTypeInt64}, InferInfoParam{ShapeVector{-2}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-2}, kNumberTypeInt64}, InferInfoParam{ShapeVector{-2}, kNumberTypeInt64}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeInt64});

    return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(Gcd, GeneralInferTest, testing::ValuesIn(prepare_params()));

}  // namespace ops
}  // namespace mindspore
