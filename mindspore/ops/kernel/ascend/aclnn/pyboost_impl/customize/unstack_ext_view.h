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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_UNSTACKEXTVIEW_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_UNSTACKEXTVIEW_

#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/auto_generate/unstack_ext_view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> UnstackExtViewAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const TensorPtr &x_tensor, const Int64ImmPtr &axis);
std::vector<tensor::TensorPtr> UnstackExtViewAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const TensorPtr &x_tensor, const int64_t &axis);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_UNSTACKEXTVIEW_
