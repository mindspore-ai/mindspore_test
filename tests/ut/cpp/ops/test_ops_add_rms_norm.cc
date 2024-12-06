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
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
    .FeedExpectedOutput({{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-5)}})
      .FeedExpectedOutput({{2, 3, 4}, {-1, -1, -1}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, 3, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-5)}})
      .FeedExpectedOutput({{2, 3, 4}, {1, 1, 1}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-5)}})
      .FeedExpectedOutput({{2, 3, 4}, {2, 1, 1}, {2, 3, 4}}, {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat16});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{-1, 4}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{2, 3, 4}, {2, 1, 1}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, -1}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{2, 3, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, 4}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{2, 3, 4}, {2, 1, 1}, {2, 3, 4}}, {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat16});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, 5}, kNumberTypeFloat16},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{-1, 5}, {1, 1}, {-1, 5}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, 5}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{-2}, {-2}, {-2}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{-2}, {-2}, {-2}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{2, 3, 4}, {1, 1, 1}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{2, 3, 4}, {2, 3, 1}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1e-6)}})
      .FeedExpectedOutput({{2, 3, 4}, {2, 3, 4}, {2, 3, 4}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AddRmsNorm, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
