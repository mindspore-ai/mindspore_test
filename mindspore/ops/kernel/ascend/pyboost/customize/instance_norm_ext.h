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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_M_T_INSTANCE_NORM_EXT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_M_T_INSTANCE_NORM_EXT_H_

#include <memory>

#include "ir/tensor.h"
#include "ir/scalar.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InstanceNormExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                    const std::optional<BaseTensorPtr> &weight,
                                    const std::optional<BaseTensorPtr> &bias,
                                    const std::optional<BaseTensorPtr> &running_mean,
                                    const std::optional<BaseTensorPtr> &running_var, const BoolImmPtr &use_input_stats,
                                    const FP32ImmPtr &momentum, const FP32ImmPtr &epsilon);

const std::optional<BaseTensorPtr> repeat_if_defined(const BaseTensorPtr &input, const int count);
ValueTuplePtr vec_to_tuple_ptr(const ShapeVector &shape);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_M_T_INSTANCE_NORM_EXT_H_
