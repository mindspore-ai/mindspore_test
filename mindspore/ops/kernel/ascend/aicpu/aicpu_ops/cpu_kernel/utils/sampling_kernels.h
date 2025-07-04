/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_UTILS_SAMPLING_KERNELS_H_
#define AICPU_UTILS_SAMPLING_KERNELS_H_

#include <stdio.h>
#include <cmath>
#include <string>
#include "cpu_context.h"
#include "utils/kernel_util.h"

namespace aicpu {
// Defines functions for different types of sampling kernels.
enum SamplingKernelType {
  // Lanczos kernel with radius 1.  Aliases but does not ring.
  Lanczos1Kernel,

  /**
   * Lanczos kernel with radius 3.  High-quality practical filter but may have
   * some ringing especially on synthetic images.
   */
  Lanczos3Kernel,

  /**
   * Lanczos kernel with radius 5.  Very-high-quality filter but may have
   * stronger ringing.
   */
  Lanczos5Kernel,

  // Gaussian kernel with radius 3, sigma = 1.5 / 3.  Less commonly used.
  GaussianKernel,

  /**
   * Rectangle function.  Equivalent to "nearest" sampling when upscaling.
   * Has value 1 in interval (-0.5, 0.5), value 0.5 on edge, and 0 elsewhere.
   */
  BoxKernel,

  /**
   * Hat/tent function with radius 1.  Equivalent to "bilinear" reconstruction
   * when upsampling.
   * Has value zero at -1.0 and 1.0.
   */
  TriangleKernel,

  /**
   * Cubic interpolant of Keys.  Equivalent to Catmull-Rom kernel.  Reasonably
   * good quality and faster than Lanczos3Kernel.
   */
  KeysCubicKernel,

  /**
   * Cubic non-interpolating scheme.  For synthetic images (especially those
   * lacking proper prefiltering), less ringing than Keys cubic kernel but less
   * sharp.
   */
  MitchellCubicKernel,

  // Always insert new kernel types before this.
  SamplingKernelTypeEnd
};

/**
 * // Converts a string into the corresponding kernel type.
 * Returns SamplingKernelTypeEnd if the string couldn't be converted.
 */
SamplingKernelType SamplingKernelTypeFromString(const std::string &str);

// The function object for a Lanczos kernel.
struct LanczosKernelFunc {
  // Pass N for LanczosN kernel.
  explicit LanczosKernelFunc(float _radius) : radius(_radius) {}
  float operator()(float x) const {
    constexpr float PI = 3.14159265359;
    auto y = std::abs(x);
    if (y > radius) {
      return 0.0;
    }
    // Need to special case the limit case of sin(x) / x when x is zero.
    if (y <= 1e-3) {
      return 1.0;
    }
    return radius * std::sin(PI * y) * std::sin(PI * y / radius) / (PI * PI * y * y);
  }
  float Radius() const { return radius; }
  const float radius;
};

struct GaussianKernelFunc {
  static constexpr float kRadiusMultiplier = 3.0f;
  /**
   * https://en.wikipedia.org/wiki/Gaussian_function
   * We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
   * for Common Resampling Tasks" for kernels with a support of 3 pixels:
   * www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
   * This implies a radius of 1.5,
   */
  explicit GaussianKernelFunc(float _radius = 1.5f) : radius(_radius), sigma(_radius / kRadiusMultiplier) {}
  float operator()(float x) const {
    x = std::abs(x);
    if (x >= radius) {
      return 0.0;
    }
    return std::exp(-x * x / (2.0 * sigma * sigma));
  }
  float Radius() const { return radius; }
  const float radius;
  // Gaussian standard deviation
  const float sigma;
};

struct BoxKernelFunc {
  float operator()(float x) const {
    x = std::abs(x);
    return x < 0.5f ? 1.f : FloatEqual(x, 0.5f) ? 0.5f : 0.0f;
  }
  float Radius() const { return 1.0f; }
};

// definition of triangle kernel
struct TriangleKernelFunc {
  float operator()(float x) const {
    x = std::abs(x);
    if (x < 1.0f) {
      return 1.0f - x;
    } else {
      return 0.0f;
    }
  }
  float Radius() const { return 1.f; }
};

// definition of cubic kernel
struct KeysCubicKernelFunc {
  float operator()(float x) const {
    x = std::abs(x);
    float res = 0.0f;
    if (x >= 2.0f) {
      res = 0.0f;
    } else if (x >= 1.0f) {
      res = ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    } else {
      res = ((1.5f * x - 2.5f) * x) * x + 1.0f;
    }
    return res;
  }
  float Radius() const { return 2.0f; }
};

struct MitchellCubicKernelFunc {
  /**
   * https://doi.org/10.1145/378456.378514
   * D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
   * graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
   * 22(4):221–228, 1988.
   */
  float operator()(float x) const {
    x = std::abs(x);
    if (x >= 2.0f) {
      return 0.0f;
    } else if (x >= 1.0f) {
      return (((-7.0f / 18.0f) * x + 2.0f) * x - 10.0f / 3.0f) * x + 16.0f / 9.0f;
    } else {
      return (((7.0f / 6.0f) * x - 2.0f) * x) * x + 8.0f / 9.0f;
    }
  }
  float Radius() const { return 2.f; }
};

inline LanczosKernelFunc CreateLanczos1Kernel() { return LanczosKernelFunc(1.0); }

inline LanczosKernelFunc CreateLanczos3Kernel() { return LanczosKernelFunc(3.0); }

inline LanczosKernelFunc CreateLanczos5Kernel() { return LanczosKernelFunc(5.0); }

inline GaussianKernelFunc CreateGaussianKernel() { return GaussianKernelFunc(1.5); }

inline BoxKernelFunc CreateBoxKernel() { return BoxKernelFunc(); }

inline TriangleKernelFunc CreateTriangleKernel() { return TriangleKernelFunc(); }

inline KeysCubicKernelFunc CreateKeysCubicKernel() { return KeysCubicKernelFunc(); }

inline MitchellCubicKernelFunc CreateMitchellCubicKernel() { return MitchellCubicKernelFunc(); }

}  // namespace aicpu

#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_SAMPLING_KERNELS_H_
