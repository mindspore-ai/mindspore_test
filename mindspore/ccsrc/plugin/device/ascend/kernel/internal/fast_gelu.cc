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

#include "plugin/device/ascend/kernel/internal/fast_gelu.h"

#include <memory>
#include "common/kernel.h"

#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalFastGeLU::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                       const internal::OutputsImmutableInfoList &outputs_ii,
                                                       const std::vector<KernelTensor *> &ms_inputs,
                                                       const std::vector<KernelTensor *> &ms_outputs) {
  return internal::CreateFastGeLUOp(inputs_ii, outputs_ii, internal::kInternalFastGeLUOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(FastGeLU, internal::kInternalFastGeLUOpName, InternalFastGeLU);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FastGeLU, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FastGeLU, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
