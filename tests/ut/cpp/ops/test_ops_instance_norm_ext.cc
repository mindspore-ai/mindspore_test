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
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)}
                    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.6)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)}
                    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)}
                    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)}
                    })
    .FeedExpectedOutput({{-1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.24)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.14)}
                    })
    .FeedExpectedOutput({{-1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{12}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{12}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{12}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{12}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.44)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.64)}
                    })
    .FeedExpectedOutput({{-1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 4, 6, 6}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.4)}
                    })
    .FeedExpectedOutput({{2, 4, 6, 6}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{12, 4, 16, 16}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.74)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.44)}
                    })
    .FeedExpectedOutput({{12, 4, 16, 16}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{12, 24, 13, 16}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{24}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{24}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool,
                                   CreateScalar<bool>(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.474)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, 
                                   CreateScalar<float>(0.424)}
                    })
    .FeedExpectedOutput({{12, 24, 13, 16}}, {kNumberTypeBFloat16});
  
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InstanceNormExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
