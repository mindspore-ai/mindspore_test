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
class GluGradCaseGenerator {
 public:
  GluGradCaseGenerator() : generator() {}
  auto cases() {
    // Only shape and type of the first operand matters. grad/dim is uncecessary.
    add_case({2}, {1}, kNumberTypeFloat64, 0);
    add_case({4, 3}, {2, 3}, kNumberTypeFloat32, 0);
    add_case({2, 3, 8}, {2, 3, 4}, kNumberTypeFloat16, 2);
    add_case({-1, -1}, {-1, -1}, kNumberTypeBFloat16, 1);
    add_case({-2}, {-2}, kNumberTypeFloat64, 4);
    return generator.Generate();
  }

 private:
  GeneralInferParamGenerator generator;
  void add_case(ShapeVector &&grad, ShapeVector &&self, TypeId dtype, int64_t dim) {
    generator
      .FeedInputArgs({
        InferInfoParam{grad, dtype},                                        // grad
        InferInfoParam{self, dtype},                                        // self
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(dim)}  // dim
      })
      .FeedExpectedOutput({self}, {dtype});
  }
};
}  // namespace

INSTANTIATE_TEST_CASE_P(GluGrad, GeneralInferTest, testing::ValuesIn(GluGradCaseGenerator().cases()));
}  // namespace mindspore::ops
