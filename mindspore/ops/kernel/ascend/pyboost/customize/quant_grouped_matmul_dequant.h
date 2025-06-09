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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_QUANT_GROUPED_MATMUL_DEQUANT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_QUANT_GROUPED_MATMUL_DEQUANT_H_

#include <memory>
#include <tuple>
#include "ir/tensor.h"
#include "ir/scalar.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void QuantGroupedMatmulDequantAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor, const BaseTensorPtr &weight_tensor,
  const BaseTensorPtr &weight_scale_tensor, const BaseTensorPtr &group_list_tensor,
  const std::optional<BaseTensorPtr> &bias_tensor, const std::optional<BaseTensorPtr> &x_scale_tensor,
  const std::optional<BaseTensorPtr> &x_offset_tensor, const std::optional<BaseTensorPtr> &smmoth_scale_tensor,
  const mindspore::StringImmPtr &x_quant_mode, const mindspore::BoolImmPtr &transpose_weight);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_QUANT_GROUPED_MATMUL_DEQUANT_H_
