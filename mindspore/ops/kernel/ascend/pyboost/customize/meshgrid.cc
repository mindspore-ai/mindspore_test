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

#include "kernel/ascend/pyboost/customize/meshgrid.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/ascend/pyboost/auto_generate/view.h"
#include "kernel/ascend/pyboost/auto_generate/broadcast_to.h"
#include "op_def/op_enum.h"
#include "mindspore/ccsrc/pyboost/customize/meshgrid.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> MeshgridAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                           const ValueTuplePtr &tensors_list,
                                                           const Int64ImmPtr &indexing) {
  MS_LOG(DEBUG) << "Meshgrid call start";
  auto outputs_list = MeshgridCustomizeCall(op, tensors_list, indexing, kAscendDevice);
  return outputs_list;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
