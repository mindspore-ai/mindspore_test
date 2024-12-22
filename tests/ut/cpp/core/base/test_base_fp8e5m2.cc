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

TEST_F(TestFloat8e5m2, ZeroConversion) {
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(0.0f))), 0.0f);
}

TEST_F(TestFloat8e5m2, OneConversion) {
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f))), 1.0f);
}

TEST_F(TestFloat8e5m2, NegativeOneConversion) {
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(-1.0f))), -1.0f);
}

TEST_F(TestFloat8e5m2, NumberConversion) {
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f))), 3.0f);
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f))), -2.5f);
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f))), 96.0f);
  EXPECT_EQ(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(1.0f))), 32768.0f);
}

TEST_F(TestFloat8e5m2, InfinityConversion) {
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isinf(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(inf)))));
}

TEST_F(TestFloat8e5m2, NaNConversion) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(nan)))));
}

}  // namespace mindspore
