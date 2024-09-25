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

#include "kernel/ascend/pyboost/customize/adaptive_avg_pool1d.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/common/pyboost/auto_generate/reshape.h"
#include "kernel/common/pyboost/auto_generate/adaptive_avg_pool2d_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int kShape2dDims = 2;
}
tensor::BaseTensorPtr AdaptiveAvgPool1DAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                       const BaseTensorPtr &input_x_tensor,
                                                       const Int64ImmPtr &output_size) {
  OpRunner::InferOpOutput(op, input_x_tensor, output_size);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto input_shape = input_x_tensor->shape();
  auto origin_shape_dim = SizeToLong(input_shape.size());

  // unsqueeze shape
  std::vector<ValuePtr> expand_input_shape;
  for (auto i = 0; i < origin_shape_dim - 1; i++) {
    expand_input_shape.emplace_back(std::make_shared<Int64Imm>(input_shape[i]));
  }
  expand_input_shape.emplace_back(std::make_shared<Int64Imm>(1));
  expand_input_shape.emplace_back(std::make_shared<Int64Imm>(input_shape[origin_shape_dim - 1]));
  auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
  auto input_x_imm = reshape_op->Call(input_x_tensor, std::make_shared<ValueTuple>(expand_input_shape));

  // call AdaptiveAvgPool2dExt
  auto adaptive_avg_pool2d_op =
    CREATE_PYBOOST_OP(AdaptiveAvgPool2DExt, op->device_context()->device_context_key_.device_name_);
  auto output_adaptive_avg_pool2d_tensor = adaptive_avg_pool2d_op->Call(
    input_x_imm, std::make_shared<ValueTuple>(std::vector<ValuePtr>({std::make_shared<Int64Imm>(1), output_size})));

  // squeeze shape
  auto shape_pool2d = output_adaptive_avg_pool2d_tensor->shape();
  auto shape_pool2d_dim = SizeToLong(shape_pool2d.size());
  std::vector<ValuePtr> squeeze_input_shape;
  if (shape_pool2d_dim <= kShape2dDims) {
    MS_LOG(EXCEPTION) << "For AdaptiveAvgPool1DAscendCustomize, the value of shape_pool2d.size is invalid.";
  }
  constexpr int offset = 2;
  for (auto i = 0; i < shape_pool2d_dim - offset; i++) {
    squeeze_input_shape.emplace_back(std::make_shared<Int64Imm>(shape_pool2d[i]));
  }
  squeeze_input_shape.emplace_back(std::make_shared<Int64Imm>(shape_pool2d[shape_pool2d_dim - 1]));
  auto output_tensor =
    reshape_op->Call(output_adaptive_avg_pool2d_tensor, std::make_shared<ValueTuple>(squeeze_input_shape));
  op->set_outputs(reshape_op->outputs());
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
