/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_MULTISCALEDEFORMABLEATTNGRAD_ASCEND_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_MULTISCALEDEFORMABLEATTNGRAD_ASCEND_H_

#include <memory>
#include <tuple>
#include "mindspore/ccsrc/pyboost/auto_generate/multi_scale_deformable_attn_grad.h"
#include "ir/tensor.h"
#include "ir/scalar.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr>
MultiScaleDeformableAttnGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &value_tensor,
                                            const BaseTensorPtr &shape_tensor, const BaseTensorPtr &offset_tensor,
                                            const BaseTensorPtr &locations_trans_tensor,
                                            const BaseTensorPtr &weight_tensor,
                                            const BaseTensorPtr &grad_output_tensor);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_
        // MULTISCALEDEFORMABLEATTNGRAD_ASCEND_H_
