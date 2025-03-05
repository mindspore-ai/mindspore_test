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

#include "kernel/gpu/pyboost/customize/meshgrid.h"
#include <memory>
#include <utility>
#include "plugin/res_manager/gpu/device/gpu_device_manager.h"
#include "mindspore/ccsrc/pyboost/customize/meshgrid.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> MeshgridGPUCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const ValueTuplePtr &tensors_list,
                                                        const Int64ImmPtr &indexing) {
  MS_LOG(DEBUG) << "Meshgrid call start";
  std::vector<tensor::BaseTensorPtr> output = MeshgridCustomizeCall(op, tensors_list, indexing, kGPUDevice);
  MS_LOG(DEBUG) << "Meshgrid call end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
