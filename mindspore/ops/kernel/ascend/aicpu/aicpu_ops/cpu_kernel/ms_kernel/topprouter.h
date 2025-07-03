/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_TOPPROUTER_H
#define AICPU_KERNELS_NORMALIZED_TOPPROUTER_H
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
template <typename T>
struct RouterInfo {
  const T *input_data;
  const int64_t capacity;
  const int64_t expert_num;
  const float threshold;
  const float *router_prob;
  const int64_t length;
  const int64_t k;
  std::vector<int64_t> expert_counter;
  std::vector<float> token_accu_weight;
  T *dispatch_index;
  T *combine_index;

  RouterInfo(const T *input_data, const int64_t capacity, const int64_t expert_num, const float threshold,
             const float *router_prob, const int64_t length, const int64_t k, std::vector<int64_t> &expert_counter,
             std::vector<float> &token_accu_weight, T *dispatch_index, T *combine_index)
      : input_data(input_data),
        capacity(capacity),
        expert_num(expert_num),
        threshold(threshold),
        router_prob(router_prob),
        length(length),
        k(k),
        expert_counter(expert_counter),
        token_accu_weight(token_accu_weight),
        dispatch_index(dispatch_index),
        combine_index(combine_index) {}
};
class TopPRouterCpuKernel : public CpuKernel {
 public:
  TopPRouterCpuKernel() = default;
  ~TopPRouterCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t TopPRouterCompute(const CpuKernelContext &ctx);
  template <typename T>
  void DoCompute(const CpuKernelContext &ctx, const int i, const int bs, const int j, RouterInfo<T> &routerinfo);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_TOPPROUTER_H
