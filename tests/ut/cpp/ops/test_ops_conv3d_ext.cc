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
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}
                  })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 4, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}
                  })
    .FeedExpectedOutput({{2, -1, -1, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 12, 124, 64, 79}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{24, 12, 4, 4, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}
                  })
    .FeedExpectedOutput({{1, 24, 121, 61, 76}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 12, 124, 64, 79}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{24, 12, 4, 4, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}
                  })
    .FeedExpectedOutput({{1, 24, 121, 61, 76}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Conv3DExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops