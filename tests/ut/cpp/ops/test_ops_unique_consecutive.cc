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
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}, {-2}, {-2}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-2}, {-2}, {-2}}, {kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1}, {-1, 2, 3}, {-1}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1}, {}, {}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{12}, {2, 2, 3}, {12}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{12}, {}, {}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)}})
    .FeedExpectedOutput({{-1, 2, 3}, {3}, {3}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)}})
    .FeedExpectedOutput({{-1, 2, 3}, {3}, {3}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{2, 2, 3}, {3}, {3}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{2, 2, 3}, {3}, {3}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)}})
    .FeedExpectedOutput({{-1, 2, 3}, {-1}, {-1}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)}})
    .FeedExpectedOutput({{-1, 2, 3}, {-1}, {-1}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, 2, 3}, {2}, {2}}, {kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)}})
    .CaseShouldThrow();
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-4)}})
    .CaseShouldThrow();
  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(UniqueConsecutive, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
