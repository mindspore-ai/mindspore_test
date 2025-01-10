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

#include "kernel/ascend/pyboost/customize/conv_transpose2d.h"
#include <memory>
#include "ir/scalar.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/auto_generate/convolution.h"
#include "kernel/common/pyboost/auto_generate/expand_dims.h"
#include "kernel/common/pyboost/auto_generate/squeeze.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/common/pyboost/op_register.h"
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

tensor::BaseTensorPtr ConvTranspose2DAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor, const BaseTensorPtr &weight_tensor,
  const std::optional<BaseTensorPtr> &bias_tensor, const ValueTuplePtr &stride, const ValueTuplePtr &padding,
  const ValueTuplePtr &output_padding, const Int64ImmPtr &groups, const ValueTuplePtr &dilation) {
  const auto &input_shape = input_tensor->shape();
  auto is_batchify = ConvNDBatchify(input_shape, 2, op->primitive()->name());

  static BoolImmPtr transposed = std::make_shared<BoolImm>(true);
  auto convolution_op = CREATE_PYBOOST_OP(Convolution, op->device_context()->device_context_key_.device_name_);
  if (is_batchify) {
    auto output_convolution = convolution_op->Call(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation,
                                                   transposed, output_padding, groups);
    op->set_outputs(convolution_op->outputs());
    return output_convolution;
  } else {
    // unsqueeze dim 0
    static auto dim = std::make_shared<Int64Imm>(0);
    auto expand_dims_op = CREATE_PYBOOST_OP(ExpandDims, op->device_context()->device_context_key_.device_name_);
    auto expand_input = expand_dims_op->Call(input_tensor, dim);
    // call convolution
    auto output_convolution = convolution_op->Call(expand_input, weight_tensor, bias_tensor, stride, padding, dilation,
                                                   transposed, output_padding, groups);
    // squeeze dim 0
    static auto squeeze_dims = std::make_shared<ValueTuple>(std::vector<ValuePtr>{dim});
    auto squeeze_op = CREATE_PYBOOST_OP(Squeeze, op->device_context()->device_context_key_.device_name_);
    auto squeeze_output_tensor = squeeze_op->Call(output_convolution, squeeze_dims);
    op->set_outputs(squeeze_op->outputs());
    return squeeze_output_tensor;
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
