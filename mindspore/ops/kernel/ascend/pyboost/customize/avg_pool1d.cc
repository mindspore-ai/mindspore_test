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

#include "kernel/ascend/pyboost/customize/avg_pool1d.h"
#include <vector>
#include <memory>

#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"

#include "kernel/ascend/pyboost/auto_generate/expand_dims.h"
#include "kernel/ascend/pyboost/auto_generate/avg_pool2d.h"
#include "kernel/common/pyboost/auto_generate/reshape.h"

#include "utils/profile.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr AvgPool1DAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                               const ValueTuplePtr &kernel_size,
                                               const std::optional<ValueTuplePtr> &stride, const ValueTuplePtr &padding,
                                               const BoolImmPtr &ceil_mode, const BoolImmPtr &count_include_pad) {
  MS_LOG(DEBUG) << "AvgPool1DAscendCustomize start";

  auto input_shape = input->shape_c();
  auto input_dim = SizeToLong(input_shape.size());

  std::vector<ValuePtr> unsqueeze_shape;
  std::vector<ValuePtr> squeeze_shape;
  for (auto i = 0; i < input_dim - 1; i++) {
    unsqueeze_shape.emplace_back(std::make_shared<Int64Imm>(input_shape[i]));
    squeeze_shape.emplace_back(std::make_shared<Int64Imm>(input_shape[i]));
  }
  unsqueeze_shape.emplace_back(std::make_shared<Int64Imm>(1));
  unsqueeze_shape.emplace_back(std::make_shared<Int64Imm>(input_shape[input_dim - 1]));
  const auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key().device_name_);
  auto expanded_input = reshape_op->Call(input, std::make_shared<ValueTuple>(unsqueeze_shape));

  auto kernel_size_val = ConvertValueTupleToVector<int64_t>(kernel_size);
  if (kernel_size_val.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, kernel_size should contain one value but got "
                             << kernel_size_val.size();
  }
  auto kernel_2d = std::make_shared<ValueTuple>(
    std::vector<ValuePtr>{std::make_shared<Int64Imm>(1), std::make_shared<Int64Imm>(kernel_size_val[0])});

  auto stride_val_tup = stride.value_or(kernel_size);
  auto stride_val = ConvertValueTupleToVector<int64_t>(stride_val_tup);
  if (stride_val.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, stride should contain one value but got " << stride_val.size();
  }
  auto stride_2d = std::make_shared<ValueTuple>(
    std::vector<ValuePtr>{std::make_shared<Int64Imm>(1), std::make_shared<Int64Imm>(stride_val[0])});

  auto padding_val = ConvertValueTupleToVector<int64_t>(padding);
  if (padding_val.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, padding should contain one value but got " << padding_val.size();
  }
  auto padding_2d = std::make_shared<ValueTuple>(
    std::vector<ValuePtr>{std::make_shared<Int64Imm>(0), std::make_shared<Int64Imm>(padding_val[0])});
  std::optional<Int64ImmPtr> divisor_override_opt = std::nullopt;

  const auto avg_pool2d_op = CREATE_PYBOOST_OP(AvgPool2D, op->device_context()->device_context_key().device_name_);
  auto avg_pool2d_output = avg_pool2d_op->Call(expanded_input, kernel_2d, stride_2d, padding_2d, ceil_mode,
                                               count_include_pad, divisor_override_opt);

  auto avg_pool2d_output_shape = avg_pool2d_output->shape_c();
  squeeze_shape.emplace_back(std::make_shared<Int64Imm>(avg_pool2d_output_shape[input_dim]));

  auto output = reshape_op->Call(avg_pool2d_output, std::make_shared<ValueTuple>(squeeze_shape));
  op->set_outputs(reshape_op->outputs());

  MS_LOG(DEBUG) << "AvgPool1DAscendCustomize end";
  return output;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
