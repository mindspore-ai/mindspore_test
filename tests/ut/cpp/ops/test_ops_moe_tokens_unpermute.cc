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

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}}
                    })
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}}
                    })
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5 * 3, 100}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{15}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{5, 3}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}}
                    })
    .FeedExpectedOutput({{5, 100}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5 * 3, 100}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{15}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{5, 3}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(50), CreateScalar<int64_t>(10)}}
                    })
    .FeedExpectedOutput({{5, 100}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5 * 3, 100}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{15}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{5, 3}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{kValueAny, kValueAny}}
                    })
    .FeedExpectedOutput({{5, 100}}, {kNumberTypeBFloat16});
  
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InnerMoeTokenUnpermute, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
