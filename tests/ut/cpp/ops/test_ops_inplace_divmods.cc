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
#include <memory>
#include "common/common_test.h"
#include "infer/ops_func_impl/inplace_divmods.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"


namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 1}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{1, 1}}, {kNumberTypeFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat64}, InferInfoParam{ShapeVector{}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt8}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt16}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt32}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt8}, InferInfoParam{ShapeVector{}, kNumberTypeInt8},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int>(1)}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(InplaceDivMods, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops