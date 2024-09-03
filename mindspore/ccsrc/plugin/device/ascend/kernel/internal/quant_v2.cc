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
#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/kernel/internal/quant_v2.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalQuantV2::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::QuantPerChannel;
  internal::ElewiseParam op_param;
  op_param.elewiseType = internal::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL;
  param_ptr->specificParam = op_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(QuantV2, InternalQuantV2);
}  // namespace kernel
}  // namespace mindspore
