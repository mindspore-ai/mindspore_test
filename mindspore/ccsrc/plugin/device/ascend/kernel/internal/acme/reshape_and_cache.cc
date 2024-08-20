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

#include "plugin/device/ascend/kernel/internal/acme/reshape_and_cache.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeReshapeAndCache::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                  const acme::OutputsImmutableInfoList &outputs_ii,
                                                  const std::vector<KernelTensor *> &ms_inputs,
                                                  const std::vector<KernelTensor *> &ms_outputs) {
  return acme::CreateReshapeAndCacheOp(inputs_ii, outputs_ii, acme::kAcmeReshapeAndCacheOpName);
}
MS_ACME_KERNEL_FACTORY_REG(ReshapeAndCache, acme::kAcmeReshapeAndCacheOpName, AcmeReshapeAndCache);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(ReshapeAndCache, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
// REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(ReshapeAndCache, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
