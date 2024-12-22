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
#include "base/hifloat8.h"

namespace mindspore {

class TestHiFloat8 : public UT::Common {
 public:
  TestHiFloat8() {}
};

TEST_F(TestHiFloat8, ZeroConversion) {
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(0.0f))), 0.0f);
}

TEST_F(TestHiFloat8, OneConversion) {
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), 1.0f);
}

TEST_F(TestHiFloat8, NegativeOneConversion) {
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(-1.0f))), -1.0f);
}

TEST_F(TestHiFloat8, NumberConversion) {
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), 0.5f);
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), -2.5f);
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), -8.0f);
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), 96.0f);
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), 768.0f);
  EXPECT_EQ(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f))), -8192.0f);
}

TEST_F(TestHiFloat8, InfinityConversion) {
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isinf(Float8_e5m2::ToFloat32(Float8_e5m2::FromRaw(Float8_e5m2::FromFloat32(inf)))));
}

TEST_F(TestHiFloat8, NaNConversion) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(HiFloat8::ToFloat32(HiFloat8::FromRaw(HiFloat8::FromFloat32(nan)))));
}

}  // namespace mindspore
