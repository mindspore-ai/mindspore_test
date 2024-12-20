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

#include <iostream>
#include <iomanip>
#include <cmath>
#include <bitset>
#define private public
#define protected public
#include "base/float8_e4m3.h"

std::string float_to_binary(float f) {
  union {
    float f;
    uint32_t i;
  } converter;
  converter.f = f;
  return std::bitset<32>(converter.i).to_string();
}

std::string format_float_binary(const std::string& bin) {
  return bin.substr(0,1) + " | " +
         bin.substr(1,8) + " | " +
         bin.substr(9,23);
}

int main() {
  float test_values[] = {
    0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
    3.14f, -2.71f, 100.0f, -100.0f,
    438.0f, 448.0f,-438.0f,
    std::numeric_limits<float>::quiet_NaN()
    // Attention that FP8 E4M3 format does not support representation of infinity (INF).
  };

  for (float f32 : test_values) {
    uint8_t float8_bits = Float8_e4m3::FromFloat32(f32);
    Float8_e4m3 float8 = Float8_e4m3::FromRaw(float8_bits);
    float f32_converted = Float8_e4m3::ToFloat32(float8);
    float diff = std::abs(f32 - f32_converted);

    std::cout << "Input float: " << f32 << std::endl;
    std::cout << "Float32 bits: " << format_float_binary(float_to_binary(f32)) << std::endl;
    std::cout << "            (S |    E    |           M           )" << std::endl;
    std::cout << "Float8_e4m3 hex: 0x" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(float8_bits) << std::dec << std::endl;
    std::cout << "Float8_e4m3 bits: " << std::bitset<8>(float8_bits).to_string().substr(0,1)
              << " | " << std::bitset<8>(float8_bits).to_string().substr(1,4)
              << " | " << std::bitset<8>(float8_bits).to_string().substr(5,3) << std::endl;
    std::cout << "               (S |  E   | M )" << std::endl;
    std::cout << "Output float: " << f32_converted << std::endl;
    std::cout << "Output bits: " << format_float_binary(float_to_binary(f32_converted)) << std::endl;
    std::cout << "Delta: " << diff << std::endl;

    const float epsilon = abs(f32)/8.0;

    if (std::isnan(f32) && std::isnan(f32_converted)) {
        std::cout << "Result: NaN test passed" << std::endl;
    } else if (std::isinf(f32) && std::isinf(f32_converted) &&
               std::signbit(f32) == std::signbit(f32_converted)) {
      std::cout << "Result: Infinity test passed" << std::endl;
    } else if (diff <= epsilon) {
      std::cout << "Result: Test passed" << std::endl;
    } else {
      std::cout << "Result: Test failed" << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
  }
  return 0;
}
