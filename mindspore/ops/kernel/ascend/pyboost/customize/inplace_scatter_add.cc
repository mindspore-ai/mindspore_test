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

#include "kernel/ascend/pyboost/customize/inplace_scatter_add.h"
#include <memory>
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "runtime/device/device_address_utils.h"
#include "ir/base_tensor.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplaceScatterAddAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                       const BaseTensorPtr &input_tensor, const Int64ImmPtr &dim,
                                                       const BaseTensorPtr &index_tensor,
                                                       const BaseTensorPtr &src_tensor) {
  const auto dim_imm = GetValue<int64_t>(dim);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index_tensor, src_tensor);
  op->set_outputs({input_tensor});
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_imm, index_tensor, src_tensor]() {
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index_tensor, src_tensor);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnScatterAdd, device_context, op->stream_id(), input_tensor, dim_imm, index_tensor, src_tensor,
                   op->output(0));
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
