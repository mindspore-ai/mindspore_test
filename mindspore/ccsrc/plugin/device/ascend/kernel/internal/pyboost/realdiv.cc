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

#include "plugin/device/ascend/kernel/internal/pyboost/realdiv.h"

#include <memory>
#include "kernel/kernel.h"

#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeKernelInfoRealDiv::CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                                    const acme::OutputsImmutableInfoList &outputs,
                                                    const std::vector<tensor::BaseTensorPtr> &ms_inputs,
                                                    const std::vector<tensor::BaseTensorPtr> &ms_outputs) {
  return acme::CreateRealDivOp(inputs, outputs, acme::kAcmeRealDivOpName);
}
MS_ACME_KERNEL_INFO_FACTORY_REG(RealDiv, acme::kAcmeRealDivOpName, AcmeKernelInfoRealDiv);
}  // namespace kernel
}  // namespace mindspore
