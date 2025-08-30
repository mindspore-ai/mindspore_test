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

#include "kernel/ascend/internal/mla_preprocess.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {

internal::InternalOpPtr InternalMlaPreprocess::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                            const internal::OutputsImmutableInfoList &outputs_ii,
                                                            const std::vector<KernelTensor *> &ms_inputs,
                                                            const std::vector<KernelTensor *> &ms_outputs) {
  internal::MlaPreprocessParam param;
  auto cache_mode = ms_inputs.at(kMlaPreprocessParamCacheModeIndex);
  if (cache_mode->dtype_id() == TypeId::kNumberTypeInt64) {
    param.n = 0;
    param.head_num = 0;
    param.cache_mode = static_cast<int32_t>(cache_mode->GetValue<int64_t>().value());
  } else {
    MS_LOG(EXCEPTION) << "MlaPreprocess [n, head_num, cache_mode]'s dtype wrong";
  }
  return internal::CreateMlaPreprocessOp(inputs_ii, outputs_ii, param, internal::kInternalMlaPreprocessOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(MlaPreprocess, internal::kInternalMlaPreprocessOpName, InternalMlaPreprocess);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MlaPreprocess, INPUT_NUM_26, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4, INDEX_5,
                                     INDEX_6, INDEX_7, INDEX_8, INDEX_9, INDEX_10, INDEX_11, INDEX_12, INDEX_13,
                                     INDEX_14, INDEX_15, INDEX_16, INDEX_17, INDEX_18, INDEX_19, INDEX_20, INDEX_21,
                                     INDEX_22, INDEX_23, INDEX_24, INDEX_25);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MlaPreprocess, OUTPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
