/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

namespace mindspore {
namespace ops {
namespace {

std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 3, 3, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 10, 10, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{3, 10, 10, 10, 10}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 3, 3, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 3, 3, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, -1, 10, -1, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{3, -1, 10, -1, 10}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AvgPool3DGradExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace ops
}  // namespace mindspore