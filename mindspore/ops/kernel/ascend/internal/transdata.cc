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

#include "kernel/ascend/internal/transdata.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalTransData::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                        const internal::OutputsImmutableInfoList &outputs_ii,
                                                        const std::vector<KernelTensor *> &ms_inputs,
                                                        const std::vector<KernelTensor *> &ms_outputs) {
  internal::TransDataParam transdata_param;
  if (ms_inputs[0]->GetStringFormat() == "FRACTAL_NZ") {
    transdata_param.transdataType = internal::TransDataParam::FRACTAL_NZ_TO_ND;
  } else {
    transdata_param.transdataType = internal::TransDataParam::ND_TO_FRACTAL_NZ;
  }
  if (primitive_->HasAttr(kAttrInternalSepcialFormat)) {
    transdata_param.specialTransdata = GetValue<int64_t>(primitive_->GetAttr(kAttrInternalSepcialFormat));
  }
  return internal::CreateTransDataOp(inputs_ii, outputs_ii, transdata_param, internal::kInternalTransDataOpName);
}

uint64_t InternalTransData::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, inputs[0]->GetStringFormat());
}

MS_INTERNAL_KERNEL_FACTORY_REG(TransData, internal::kInternalTransDataOpName, InternalTransData);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(TransData, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(TransData, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
