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

#include "mindspore/ops/kernel/common/pyboost/customize/reshape.h"
#include <string>
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/common/pyboost/auto_generate/copy.h"
#include "kernel/common/pyboost/auto_generate/view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                       const ValueTuplePtr &shape, const std::string &device_target) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto view_op = CREATE_PYBOOST_OP(View, device_target);
  auto old_storage_info = input_tensor->storage_info();
  if (old_storage_info == nullptr || old_storage_info->is_contiguous) {
    // Tensor is contiguous, reshape by view
    auto output_tensor = view_op->Call(input_tensor, shape);
    op->set_outputs(view_op->outputs());
    return output_tensor;
  }

  // Tensor is not contiguous, need call copy first
  auto copy_op = CREATE_PYBOOST_OP(Copy, device_target);
  copy_op->set_stream_id(op->stream_id());
  const auto copy_tensor = copy_op->Call(input_tensor);
  auto output_tensor = view_op->Call(copy_tensor, shape);
  op->set_outputs(view_op->outputs());
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
