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

#include "inc/cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "topprouter.h"
#include <vector>

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 6;
constexpr size_t kXIndex = 0;
constexpr size_t kCapacityIndex = 1;
constexpr size_t kExpertNumIndex = 2;
constexpr size_t kDropTypeIndex = 3;
constexpr size_t kThresholdIndex = 4;
constexpr size_t kRouterProbIndex = 5;
constexpr size_t kDispatchIndex = 0;
constexpr size_t kCombineIndex = 1;

const char *const kTopPRouter = "TopPRouter";
#define TOPPROUTER_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                      \
    uint32_t result = TopPRouterCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "TopPRouter kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }

}  // namespace

namespace aicpu {
uint32_t TopPRouterCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "TopPRouter check input and output number failed.");
  auto output_type = ctx.Output(0)->GetDataType();
  switch (output_type) {
    TOPPROUTER_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TOPPROUTER_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Output data type [%s] not support.", DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopPRouterCpuKernel::TopPRouterCompute(const CpuKernelContext &ctx) {
  auto input_data = static_cast<T *>(ctx.Input(kXIndex)->GetData());
  auto capacity_ptr = static_cast<int64_t *>(ctx.Input(kCapacityIndex)->GetData());
  auto expert_num_ptr = static_cast<int64_t *>(ctx.Input(kExpertNumIndex)->GetData());
  auto drop_type_ptr = static_cast<int64_t *>(ctx.Input(kDropTypeIndex)->GetData());
  auto threshold_ptr = static_cast<float *>(ctx.Input(kThresholdIndex)->GetData());
  auto router_prob = static_cast<float *>(ctx.Input(kRouterProbIndex)->GetData());

  auto dispatch_index = static_cast<T *>(ctx.Output(kDispatchIndex)->GetData());
  auto combine_index = static_cast<T *>(ctx.Output(kCombineIndex)->GetData());

  auto input_shape = ctx.Input(kXIndex)->GetTensorShape();
  auto batch = input_shape->GetDimSize(kXIndex);
  auto length = input_shape->GetDimSize(kCapacityIndex);
  auto k = input_shape->GetDimSize(kExpertNumIndex);
  auto expert_num = *expert_num_ptr;
  auto drop_type = *drop_type_ptr;
  auto capacity = *capacity_ptr;
  auto threshold = *threshold_ptr;

  // init dispatch index
  auto dispatch_shape = ctx.Output(kDispatchIndex)->GetTensorShape();
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
void TopPRouterCpuKernel::DoCompute(const CpuKernelContext &ctx, const int i, const int bs, const int j,
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
  auto input_index = bs * length * k + i * k + j;
  auto prob_index = bs * length + i;
  auto expert_id = input_data[input_index];
  auto cnt_index = bs * expert_num + expert_id;
  auto position_in_expert = expert_counter[cnt_index];
  bool condition =
    (position_in_expert < capacity && token_accu_weight[prob_index] + router_prob[input_index] < threshold);
  if (condition) {
    dispatch_index[bs * expert_num * capacity + expert_id * capacity + position_in_expert] =
      static_cast<T>(token_index + 1);
    combine_index[input_index] = static_cast<T>(expert_id * (capacity + 1) + position_in_expert + 1);
    expert_counter[cnt_index] = static_cast<T>(position_in_expert + 1);
  } else {
    combine_index[input_index] = static_cast<T>(expert_id * (capacity + 1));
  }
  token_accu_weight[prob_index] = token_accu_weight[prob_index] + router_prob[input_index];
}

REGISTER_MS_CPU_KERNEL(kTopPRouter, TopPRouterCpuKernel);

}  // namespace aicpu
