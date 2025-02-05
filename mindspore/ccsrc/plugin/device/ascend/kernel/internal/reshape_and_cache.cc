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

#include "plugin/device/ascend/kernel/internal/reshape_and_cache.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalReshapeAndCache::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                              const internal::OutputsImmutableInfoList &outputs_ii,
                                                              const std::vector<KernelTensor *> &ms_inputs,
                                                              const std::vector<KernelTensor *> &ms_outputs) {
  internal::ReshapeAndCacheParam param;
  auto isPrefill = ms_inputs.at(kIndex9);
  if (isPrefill->dtype_id() == TypeId::kNumberTypeBool) {
    param.is_prefill = static_cast<bool>(isPrefill->GetValue<bool>().value());
  } else {
    MS_LOG(EXCEPTION) << "ReshapAndCache input[9] dtype is not kNumberTypeBool";
  }

  auto cacheConfig = ms_inputs.at(kIndex10);
  if (cacheConfig->dtype_id() == TypeId::kNumberTypeInt64) {
    param.cache_config = static_cast<int32_t>(cacheConfig->GetValue<int64_t>().value());
  } else {
    MS_LOG(EXCEPTION) << "ReshapAndCache input[10] dtype is not kNumberTypeInt32";
  }
  return internal::CreateReshapeAndCacheOp(inputs_ii, outputs_ii, param, internal::kInternalReshapeAndCacheOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(ReshapeAndCache, internal::kInternalReshapeAndCacheOpName, InternalReshapeAndCache);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(ReshapeAndCache, INPUT_NUM_11, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4,
                                     INDEX_5, INDEX_6, INDEX_7, INDEX_8, INDEX_9, INDEX_10);
}  // namespace kernel
}  // namespace mindspore
