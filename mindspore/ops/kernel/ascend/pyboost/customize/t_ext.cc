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

#include "kernel/ascend/pyboost/customize/t_ext.h"
#include <memory>
#include <vector>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/ascend/pyboost/auto_generate/transpose.h"
#include "mindspore/ccsrc/pyboost/auto_generate/copy.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void TExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "TExt Launch start";
  OpRunner::InferOpOutput(op, input_tensor);
  auto device_context = op->device_context();
  auto input_rank = input_tensor->shape().size();

  const auto &device_name = device_context->device_context_key_.device_name_;
  auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
  std::vector<ValuePtr> perm(input_rank);
  for (size_t i = 0; i < input_rank; ++i) {
    perm[i] = MakeValue(static_cast<int64_t>(input_rank - i - 1));
  }
  auto output_tensor = transpose_op->Call(input_tensor, std::make_shared<ValueTuple>(perm));
  op->set_outputs({output_tensor});

  MS_LOG(DEBUG) << "TExt Launch end";
  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
