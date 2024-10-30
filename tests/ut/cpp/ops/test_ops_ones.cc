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

#define I64(x) (static_cast<int64_t>((x)))
namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat32)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3), CreateScalar<int64_t>(3)}},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt64)}})
    .FeedExpectedOutput({{2, 3, 3}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{kValueAny, CreateScalar<int64_t>(3), CreateScalar<int64_t>(3)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt64)}})
    .FeedExpectedOutput({{-1, 3, 3}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt32)}})
    .FeedExpectedOutput({{}}, {kNumberTypeInt32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Ones, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
