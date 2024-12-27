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
#include <memory>
#include <algorithm>
#include <string>
#include "kernel/ascend/pyboost/customize/conv3d_ext.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/auto_generate/convolution.h"
#include "mindspore/ccsrc/pyboost/auto_generate/reshape.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
bool ConvNDBatchify(const ShapeVector &input_shape, const int64_t num_spatial_dims, const std::string &func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  auto origin_shape_dim = SizeToLong(input_shape.size());
  const auto is_batched = (origin_shape_dim == dim_count_batch);
  if (origin_shape_dim != dim_count_no_batch && !is_batched) {
    MS_LOG(EXCEPTION) << "Expected " << dim_count_no_batch << "D (unbatched) or " << dim_count_batch
                      << "D (batched) input to " << func_name << ", but got input of size: " << origin_shape_dim;
  }
  return is_batched;
}
}  // namespace

tensor::BaseTensorPtr Conv3DExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                               const BaseTensorPtr &weight_tensor,
                                               const std::optional<BaseTensorPtr> &bias_tensor,
                                               const ValueTuplePtr &stride, const ValueTuplePtr &pad,
                                               const ValueTuplePtr &dilation, const Int64ImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, bias_tensor, stride, pad, dilation, group);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_tensor, bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  auto input_shape = input_tensor->shape();
  auto is_batchify = ConvNDBatchify(input_shape, 3, "conv3d");

  BoolImmPtr transposed_imm = std::make_shared<BoolImm>(false);
  ValueTuplePtr output_padding_vector_imm = std::make_shared<ValueTuple>(std::vector<ValuePtr>(
    {std::make_shared<Int64Imm>(0), std::make_shared<Int64Imm>(0), std::make_shared<Int64Imm>(0)}));

  auto convolution_op = CREATE_PYBOOST_OP(Convolution, op->device_context()->device_context_key_.device_name_);
  if (is_batchify) {
    auto output_imm = convolution_op->Call(input_tensor, weight_tensor, bias_tensor, stride, pad, dilation,
                                           transposed_imm, output_padding_vector_imm, group);
    op->set_outputs(convolution_op->outputs());
    return output_imm;
  } else {
    std::vector<ValuePtr> expand_input_shape;
    expand_input_shape.insert(expand_input_shape.begin(), std::make_shared<Int64Imm>(1));
    std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(expand_input_shape),
                   [](int64_t e) { return std::make_shared<Int64Imm>(e); });

    auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
    auto expand_input_x_imm = reshape_op->Call(input_tensor, std::make_shared<ValueTuple>(expand_input_shape));

    auto output_imm = convolution_op->Call(expand_input_x_imm, weight_tensor, bias_tensor, stride, pad, dilation,
                                           transposed_imm, output_padding_vector_imm, group);

    auto output_imm_shape = output_imm->shape();
    std::vector<ValuePtr> squeeze_output_shape;
    for (int64_t i = 1; i < SizeToLong(output_imm_shape.size()); i++) {
      squeeze_output_shape.emplace_back(std::make_shared<Int64Imm>(output_imm_shape[i]));
    }
    auto squeeze_output_tensor = reshape_op->Call(output_imm, std::make_shared<ValueTuple>(squeeze_output_shape));
    op->set_outputs(reshape_op->outputs());
    return squeeze_output_tensor;
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
