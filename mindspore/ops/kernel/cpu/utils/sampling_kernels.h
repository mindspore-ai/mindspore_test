/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/sampling_kernels.h

Additional modifications made by Huawei Technologies Co., Ltd in 2020-2022.
==============================================================================*/

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_

#include <cmath>
#include <limits>
#include <string>

namespace mindspore {
namespace kernel {
enum KernelType { Lanczos1, Lanczos3, Lanczos5, Gaussian, Box, Triangle, KeysCubic, MitchellCubic, TypeEnd };
KernelType GetSamplingKernelType(const std::string &str);
static constexpr float kRValue0 = 0.0f;
static constexpr float kRValue1 = 1.0f;
static constexpr float kRValue2 = 2.0f;

struct ComputerLanczosKernel {
  explicit ComputerLanczosKernel(float _radius) : radius(_radius) {}
  float operator()(float distance) const {
    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr float kNearZero = 1e-3f;
    const float abs_distance = std::abs(distance);
    if (abs_distance > radius) {
      return kRValue0;
    }
    if (abs_distance <= kNearZero) {
      return kRValue1;
    }
    const float sin_pi_x = std::sin(kPi * abs_distance);
    const float sin_pi_x_over_r = std::sin(kPi * abs_distance / radius);
    const float denom = kPi * kPi * abs_distance * abs_distance;
    return (radius * sin_pi_x * sin_pi_x_over_r) / denom;
  }
  float Radius() const noexcept { return radius; }
  const float radius;
};

struct ComputerGaussianKernel {
  static constexpr float kRadiusMultiplier = 3.0f;

  explicit ComputerGaussianKernel(float _radius = 1.5f)
      : radius(_radius), sigma(_radius / kRadiusMultiplier), inv_two_sigma_squared(1.0f / (2.0f * sigma * sigma)) {}
  float operator()(float distance) const {
    const float abs_distance = std::abs(distance);
    if (abs_distance >= radius) {
      return kRValue0;
    }
    const float squared = abs_distance * abs_distance;
    return std::exp(-squared * inv_two_sigma_squared);
  }
  float Radius() const noexcept { return radius; }
  const float radius;
  const float sigma;
  const float inv_two_sigma_squared;
};

struct ComputerBoxKernel {
  float operator()(float input) const {
    float result;
    input = std::abs(input);
    if (input < 0.5f) {
      result = kRValue1;
    } else if (std::fabs(input - 0.5f) <= std::numeric_limits<float>::epsilon()) {
      result = 0.5f;
    } else {
      result = kRValue0;
    }
    return result;
  }
  float Radius() const noexcept { return kRValue1; }
};

struct ComputetTriangleKernel {
  float operator()(float input) const {
    float result;
    input = std::abs(input);
    if (input < kRValue1) {
      result = kRValue1 - input;
    } else {
      result = kRValue0;
    }
    return result;
  }
  float Radius() const noexcept { return kRValue1; }
};

struct ComputerKeysCubicKernel {
  float operator()(float input) const {
    input = std::abs(input);
    float result;
    if (input >= kRValue2) {
      result = kRValue0;
    } else if (input >= kRValue1) {
      result = -0.5f * input + 2.5f;
      result = result * input - 4.0f;
      result = result * input + kRValue2;
    } else {
      result = (1.5f * input - 2.5f) * input;
      result = result * input + kRValue1;
    }
    return result;
  }
  float Radius() const noexcept { return kRValue2; }
};

struct ComputerMitchellCubicKernel {
  float operator()(float distance) const {
    const float abs_distance = std::abs(distance);
    if (abs_distance >= kRValue2) {
      return kRValue0;
    }
    if (abs_distance >= kRValue1) {
      return (((-7.0f / 18.0f) * abs_distance + kRValue2) * abs_distance - 10.0f / 3.0f) * abs_distance + 16.0f / 9.0f;
    }
    return (((7.0f / 6.0f) * abs_distance - kRValue2) * abs_distance) * abs_distance + 8.0f / 9.0f;
  }
  float Radius() const noexcept { return kRValue2; }
};

inline ComputerLanczosKernel CreateLanczos1Kernel() { return ComputerLanczosKernel(1.0f); }

inline ComputerLanczosKernel CreateLanczos3Kernel() { return ComputerLanczosKernel(3.0f); }

inline ComputerLanczosKernel CreateLanczos5Kernel() { return ComputerLanczosKernel(5.0f); }

inline ComputerGaussianKernel CreateGaussianKernel() { return ComputerGaussianKernel(1.5f); }

inline ComputerBoxKernel CreateBoxKernel() { return ComputerBoxKernel(); }

inline ComputetTriangleKernel CreateTriangleKernel() { return ComputetTriangleKernel(); }

inline ComputerKeysCubicKernel CreateKeysCubicKernel() { return ComputerKeysCubicKernel(); }

inline ComputerMitchellCubicKernel CreateMitchellCubicKernel() { return ComputerMitchellCubicKernel(); }
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_
