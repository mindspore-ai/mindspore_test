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
#include "ir/anf.h"
#include "ir/base_tensor.h"
#include "ir/dtype/number.h"
#include "infer/ops_func_impl/avg_pool1d.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore {
namespace ops {
namespace {

std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3, 10, 3}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3, 10, 9}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{-1, -1, 2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{kValueAny}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3, 10, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{kValueAny}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3, 10, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10, 10}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(4)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                      ValuePtrList{kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3, 10, -1}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AvgPool1D, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace ops
}  // namespace mindspore