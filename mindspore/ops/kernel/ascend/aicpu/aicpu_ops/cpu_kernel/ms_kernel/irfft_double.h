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
#ifndef AICPU_KERNELS_NORMALIZED_IRFFT_DOUBLE_H_
#define AICPU_KERNELS_NORMALIZED_IRFFT_DOUBLE_H_

#include <vector>
#include <numeric>
#include <complex>
#include <securec.h>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class IRFFTDoubleCpuKernel : public CpuKernel {
 public:
  ~IRFFTDoubleCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t IRFFTDoubleCompute(CpuKernelContext &ctx);

  std::string op_name_;
};
}  // namespace aicpu
#endif  //  AICPU_KERNELS_NORMALIZED_IRFFT_DOUBLE_H_
