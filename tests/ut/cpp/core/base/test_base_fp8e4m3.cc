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
#include <cmath>
#include <limits>
#include "common/common_test.h"
#define private public
#include "base/fp8_e4m3.h"

namespace mindspore {

class TestFloat8e4m3 : public UT::Common {
 public:
  TestFloat8e4m3() {}
};

/// Feature: Data format conversion in Float8_e4m3.
/// Description: ZeroConversion between Float8_e4m3 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e4m3, ZeroConversion) {
  Float8_e4m3 num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(0.0f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e4m3.
/// Description: OneConversion between Float8_e4m3 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e4m3, OneConversion) {
  Float8_e4m3 num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(1.0f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e4m3.
/// Description: NegativeOneConversion between Float8_e4m3 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e4m3, NegativeOneConversion) {
  Float8_e4m3 num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(-1.0f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e4m3.
/// Description: NumberConversion between Float8_e4m3 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e4m3, NumberConversion) {
  Float8_e4m3 num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(0.5f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
  num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(3.25f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
  num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(-2.75f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
  num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(96.0f));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
  num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(448.0f));
}

/// Feature: Data format conversion in Float8_e4m3.
/// Description: NaNConversion between Float8_e4m3 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e4m3, NaNConversion) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  Float8_e4m3 num = Float8_e4m3::FromRaw(Float8_e4m3::FromFloat32(nan));
  EXPECT_TRUE(Float8_e4m3::FromFloat32(Float8_e4m3::ToFloat32(num)) == num);
}

}  // namespace mindspore
