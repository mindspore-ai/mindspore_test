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

#include "common/common_test.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/inplace_index_put.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 4, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeArray{{2, 2, 2}, {2, 2}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{2, 2, 2}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, 4, 3}}, {kNumberTypeUInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{2, 2}, {1, 2}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{2, 2}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{4, 2, 3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{10, 4, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeArray{{2, 3}, {3, 1, 1}, {3}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{10, 4, 2}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, 3, -1}, kNumberTypeInt8},
                    InferInfoParam{ShapeArray{{2, -1}, {3, 1, 2}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{5, 3, -1}}, {kNumberTypeInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 1, -1}, kNumberTypeInt64},
                    InferInfoParam{ShapeArray{{2, 3}, {3, 1, 1}, {3}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{1, 1, -1}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeArray{{2, 3}, {3, 1, 1}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{1, 1, -1}}, {kNumberTypeBFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeInt16},
                    InferInfoParam{ShapeArray{{2, 3}, {3, 1, 1}, {3}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 1, -1, 9}, kNumberTypeInt64},
                    InferInfoParam{ShapeArray{{-2}, {3, 1, 1}, {3}}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{4, 1, -1, 9}}, {kNumberTypeInt64});

  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(InplaceIndexPut, GeneralInferTest, testing::ValuesIn(prepare_params()));

}  // namespace ops
}  // namespace mindspore
