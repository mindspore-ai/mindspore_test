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

#include "kernel/ascend/pyboost/customize/flatten_ext.h"
#include <memory>
#include <algorithm>
#include <functional>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/auto_generate/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void FlattenExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x_tensor,
                               const Int64ImmPtr &start_dim, const Int64ImmPtr &end_dim) {
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
  OpRunner::InferOpOutput(op, input_x_tensor, start_dim, end_dim);
  const ShapeVector &output_shape = op->output_value_simple_info()->shape_vector_[0];
  auto new_shape = output_shape;
  auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
  reshape_op->Call(input_x_tensor, new_shape);
  op->set_outputs(reshape_op->outputs());
  MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
