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
#include "base/float8_e5m2.h"

namespace mindspore {

class TestFloat8e5m2 : public UT::Common {
 public:
  TestFloat8e5m2() {}
};

/// Feature: Data format conversion in Float8_e5m2.
/// Description: ZeroConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, ZeroConversion) {
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(0.0f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e5m2.
/// Description: OneConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, OneConversion) {
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e5m2.
/// Description: NegativeOneConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, NegativeOneConversion) {
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(-1.0f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e5m2.
/// Description: NumberConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, NumberConversion) {
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(3.0f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
  num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(-2.5f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
  num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(96.0f));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
  num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(32768.0f));
}

/// Feature: Data format conversion in Float8_e5m2.
/// Description: InfinityConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, InfinityConversion) {
  float inf = std::numeric_limits<float>::infinity();
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(inf));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
}

/// Feature: Data format conversion in Float8_e5m2.
/// Description: NaNConversion between Float8_e5m2 and Float32.
/// Expectation: No exception.
TEST_F(TestFloat8e5m2, NaNConversion) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  Float8_e5m2 num = Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(nan));
  EXPECT_TRUE(Float8_e5m2::FromFloat32(Float8_e5m2::ToFloat32(num)) == num);
}

}  // namespace mindspore
