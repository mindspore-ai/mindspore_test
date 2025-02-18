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

#include "inc/cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "topkrouter.h"
#include <vector>

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 6;
const char *const kTopKRouter = "TopKRouter";
#define TOPKROUTER_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                      \
    uint32_t result = TopKRouterCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "TopKRouter kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }

}  // namespace

namespace aicpu {
uint32_t TopKRouterCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "TopKRouter check input and output number failed.");
  auto output_type = ctx.Output(0)->GetDataType();
  switch (output_type) {
    TOPKROUTER_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TOPKROUTER_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Output data type [%s] not support.", DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKRouterCpuKernel::TopKRouterCompute(const CpuKernelContext &ctx) {
  auto input_data = static_cast<T *>(ctx.Input(0)->GetData());
  auto capacity_ptr = static_cast<int64_t *>(ctx.Input(1)->GetData());
  auto expert_num_ptr = static_cast<int64_t *>(ctx.Input(2)->GetData());
  auto drop_type_ptr = static_cast<int64_t *>(ctx.Input(3)->GetData());
  auto threshold_ptr = static_cast<float *>(ctx.Input(4)->GetData());
  auto router_prob = static_cast<float *>(ctx.Input(5)->GetData());

  auto dispatch_index = static_cast<T *>(ctx.Output(0)->GetData());
  auto combine_index = static_cast<T *>(ctx.Output(1)->GetData());

  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto batch = input_shape->GetDimSize(0);
  auto length = input_shape->GetDimSize(1);
  auto k = input_shape->GetDimSize(2);
  auto expert_num = *expert_num_ptr;
  auto drop_type = *drop_type_ptr;
  auto capacity = *capacity_ptr;
  auto threshold = *threshold_ptr;

  // init dispatch index
  auto dispatch_shape = ctx.Output(0)->GetTensorShape();
  auto dispatch_num = dispatch_shape->NumElements();
  for (int i = 0; i < dispatch_num; i++) {
    dispatch_index[i] = 0;
  }
  // init counter
  std::vector<int64_t> expert_counter(batch * expert_num, 0);
  std::vector<float> token_accu_weight(batch * length, 0);

  RouterInfo<T> routerinfo(input_data, capacity, expert_num, threshold, router_prob, length, k, expert_counter,
                           token_accu_weight, dispatch_index, combine_index);

  if (drop_type == 0) {
    for (int bs = 0; bs < batch; bs++) {
      for (int i = 0; i < length; i++) {
        for (int j = 0; j < k; j++) {
          DoCompute(ctx, i, bs, j, routerinfo);
        }
      }
    }
  } else {
    for (int bs = 0; bs < batch; bs++) {
      for (int j = 0; j < k; j++) {
        for (int i = 0; i < length; i++) {
          DoCompute(ctx, i, bs, j, routerinfo);
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void TopKRouterCpuKernel::DoCompute(const CpuKernelContext &ctx, const int i, const int bs, const int j,
                                    RouterInfo<T> &routerinfo) {
  const T *input_data = routerinfo.input_data;
  const int64_t capacity = routerinfo.capacity;
  const int64_t expert_num = routerinfo.expert_num;
  const float threshold = routerinfo.threshold;
  const float *router_prob = routerinfo.router_prob;
  const int64_t length = routerinfo.length;
  const int64_t k = routerinfo.k;

  std::vector<int64_t> &expert_counter = routerinfo.expert_counter;
  std::vector<float> &token_accu_weight = routerinfo.token_accu_weight;

  T *dispatch_index = routerinfo.dispatch_index;
  T *combine_index = routerinfo.combine_index;

  // add
  auto token_index = i;
  auto expert_id = input_data[bs * length * k + i * k + j];
  auto position_in_expert = expert_counter[bs * expert_num + expert_id];
  bool condition;
  if (threshold > 0) {
    condition = (position_in_expert < capacity &&
                 token_accu_weight[bs * length + i] + router_prob[bs * length * k + i * k + j] < threshold);
  } else {
    condition = (position_in_expert < capacity);
  }
  if (condition) {
    dispatch_index[bs * expert_num * capacity + expert_id * capacity + position_in_expert] =
      static_cast<T>(token_index + 1);
    combine_index[bs * length * k + i * k + j] = static_cast<T>(expert_id * (capacity + 1) + position_in_expert + 1);
    expert_counter[bs * expert_num + expert_id] = static_cast<T>(position_in_expert + 1);
  } else {
    combine_index[bs * length * k + i * k + j] = static_cast<T>(expert_id * (capacity + 1));
    if (threshold > 0) {
      token_accu_weight[bs * length + i] =
        token_accu_weight[bs * length + i] + router_prob[bs * length * k + i * k + j];
    }
  }
}

REGISTER_MS_CPU_KERNEL(kTopKRouter, TopKRouterCpuKernel);

}  // namespace aicpu
