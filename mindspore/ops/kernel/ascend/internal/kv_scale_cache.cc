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

#include "kernel/ascend/internal/kv_scale_cache.h"
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKvScaleCache::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                           const internal::OutputsImmutableInfoList &outputs_ii,
                                                           const std::vector<KernelTensor *> &ms_inputs,
                                                           const std::vector<KernelTensor *> &ms_outputs) {
  internal::KvScaleCacheParam param;
  param.cache_mode = static_cast<int32_t>(ms_inputs[kIndex4]->GetValue<int64_t>().value());
  MS_LOG(INFO) << "Create kernel: " << internal::kInternalKvScaleCacheOpName << " cache_mode: " << param.cache_mode;
  return internal::CreateKvScaleCacheOp(inputs_ii, outputs_ii, param, internal::kInternalKvScaleCacheOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(KvScaleCache, internal::kInternalKvScaleCacheOpName, InternalKvScaleCache);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(KvScaleCache, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_2);

}  // namespace kernel
}  // namespace mindspore
