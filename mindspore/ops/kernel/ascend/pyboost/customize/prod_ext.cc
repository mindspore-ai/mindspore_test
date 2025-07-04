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

#include "kernel/ascend/pyboost/customize/prod_ext.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ProdExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                         const std::optional<Int64ImmPtr> &axis, const BoolImmPtr &keep_dims,
                                         const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, input_tensor, axis, keep_dims, dtype);

  int64_t axis_imm = 0;
  bool keep_dims_imm = false;
  bool is_all_reduce = false;
  if (axis.has_value()) {
    axis_imm = GetValue<int64_t>(axis.value());
    keep_dims_imm = GetValue<bool>(keep_dims);
  } else {
    is_all_reduce = true;
  }

  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = op->output_value_simple_info()->dtype_vector_[kIndex0]->type_id();

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  if (is_all_reduce) {
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, out_dtype]() {
      auto device_context = op->device_context();

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnProd, device_context, op->stream_id(), input_tensor, out_dtype, op->output(0));
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  } else {
    PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis_imm, keep_dims_imm, out_dtype]() {
        auto device_context = op->device_context();

        PyBoostUtils::MallocOpInputs(device_context, input_tensor);
        PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

        MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
        LAUNCH_ACLNN(aclnnProdDim, device_context, op->stream_id(), input_tensor, axis_imm, keep_dims_imm, out_dtype,
                     op->output(0));
        MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
      }));
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
