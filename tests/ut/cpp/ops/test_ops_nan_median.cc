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
class NanMedianCaseGenerator {
 public:
  NanMedianCaseGenerator() : generator() {}
  auto cases() {
    // Always returns Scalar[input.dtype]
    add_case({}, kNumberTypeFloat16);
    add_case({2, 3, 4}, kNumberTypeFloat32);
    add_case({2, 4, -1}, kNumberTypeFloat64);
    add_case({-1, -1, -1}, kNumberTypeBFloat16);
    add_case({-2}, kNumberTypeFloat16);
    return generator.Generate();
  }

 private:
  GeneralInferParamGenerator generator;
  void add_case(ShapeVector shape, TypeId dtype) {
    generator.FeedInputArgs({InferInfoParam{shape, dtype}}).FeedExpectedOutput({{}}, {dtype});
  }
};
}  // namespace

INSTANTIATE_TEST_CASE_P(NanMedian, GeneralInferTest, testing::ValuesIn(NanMedianCaseGenerator().cases()));
}  // namespace mindspore::ops
