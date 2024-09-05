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

#include "plugin/device/ascend/kernel/internal/acme/transdata.h"

#include <memory>
#include "kernel/kernel.h"

#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeTransData::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                            const acme::OutputsImmutableInfoList &outputs_ii,
                                            const std::vector<KernelTensor *> &ms_inputs,
                                            const std::vector<KernelTensor *> &ms_outputs) {
  acme::TransDataParam transdata_param;
  if (ms_inputs[0]->GetStringFormat() == "FRACTAL_NZ") {
    transdata_param.transdataType = acme::TransDataParam::FRACTAL_NZ_TO_ND;
  } else {
    transdata_param.transdataType = acme::TransDataParam::ND_TO_FRACTAL_NZ;
  }
  if (primitive_->HasAttr(kAttrInternalSepcialFormat)) {
    transdata_param.specialTransdata = GetValue<int64_t>(primitive_->GetAttr(kAttrInternalSepcialFormat));
  }
  return acme::CreateTransDataOp(inputs_ii, outputs_ii, transdata_param, acme::kAcmeTransDataOpName);
}

uint64_t AcmeTransData::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  return AcmeTilingCache::GenerateKey(kernel_name_, inputs, inputs[0]->GetStringFormat());
}

MS_ACME_KERNEL_FACTORY_REG(TransData, acme::kAcmeTransDataOpName, AcmeTransData);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(TransData, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(TransData, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
