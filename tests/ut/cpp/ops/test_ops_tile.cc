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
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{3, 4}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}}})
    .FeedExpectedOutput({{2, 6, 8}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}}})
    .FeedExpectedOutput({{4, 6, 8}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-1, 3, -1}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}}})
    .FeedExpectedOutput({{-1, 6, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                                   ValuePtrList{kValueAny, CreateScalar<int64_t>(2), kValueAny}}})
    .FeedExpectedOutput({{-1, 6, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}, {}}, kNumberTypeInt64}})
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 3, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), kValueAny, kValueAny}}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}, true}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}, true}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3), CreateScalar<int64_t>(4)}}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                                   ValuePtrList{kValueAny, CreateScalar<int64_t>(3), kValueAny}}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}}})
    .FeedExpectedOutput({{2, 6, 8}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-1, 3, -1}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(2), kValueAny}}})
    .FeedExpectedOutput({{-1, 6, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32}, InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2, -1, -1}}, {kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Tile, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
