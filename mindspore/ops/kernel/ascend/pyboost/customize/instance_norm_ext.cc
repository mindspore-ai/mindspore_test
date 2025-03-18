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

#include "kernel/ascend/pyboost/customize/instance_norm_ext.h"
#include <memory>
#include <vector>
#include <functional>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::optional<BaseTensorPtr> repeat_if_defined(const std::optional<BaseTensorPtr> &input, const int count) {
  if (!input.has_value()) {
    return std::nullopt;
  }
  std::vector<ValuePtr> cnt_vec;
  cnt_vec.emplace_back(std::make_shared<Int64Imm>(count));
  auto cnt_tuple = std::make_shared<ValueTuple>(cnt_vec);
  return repeat(input.value(), cnt_tuple);
}

ValueTuplePtr vec_to_tuple_ptr(const ShapeVector &shape) {
  std::vector<ValuePtr> input_shape_value;
  for (auto i = 0; i < SizeToLong(shape.size()); i++) {
    input_shape_value.emplace_back(std::make_shared<Int64Imm>(shape[i]));
  }
  return std::make_shared<ValueTuple>(input_shape_value);
}

void InstanceNormExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                    const std::optional<BaseTensorPtr> &weight,
                                    const std::optional<BaseTensorPtr> &bias,
                                    const std::optional<BaseTensorPtr> &running_mean,
                                    const std::optional<BaseTensorPtr> &running_var, const BoolImmPtr &use_input_stats,
                                    const FP32ImmPtr &momentum, const FP32ImmPtr &epsilon) {
  // Convert ValuePtr to c++ scalar
  MS_LOG(DEBUG) << "InstanceNormExt Launch start.";
  OpRunner::InferOpOutput(op, input_tensor, weight, bias, running_mean, running_var, use_input_stats, momentum,
                          epsilon);
  auto input_shape = input_tensor->shape();
  auto input_shape_ori = input_shape;
  auto b = input_shape[0];
  auto c = input_shape[1];
  input_shape[1] = b * c;
  input_shape[0] = 1;

  auto weight_ = repeat_if_defined(weight, b);
  auto bias_ = repeat_if_defined(bias, b);
  auto running_mean_ = repeat_if_defined(running_mean, b);
  auto running_var_ = repeat_if_defined(running_var, b);

  auto input_shape_tuple_ptr = vec_to_tuple_ptr(input_shape);
  auto con_input = contiguous(input_tensor);
  auto input_reshaped = view(con_input, input_shape_tuple_ptr);

  auto out_ =
    batch_norm_ext(input_reshaped, weight_, bias_, running_mean_, running_var_, use_input_stats, momentum, epsilon);
  auto out_bn = std::get<0>(out_);

  if (running_mean.has_value() && running_var.has_value()) {
    auto running_shape = vec_to_tuple_ptr({b, c});
    running_mean_ = view(running_mean_.value(), running_shape);
    running_var_ = view(running_var_.value(), running_shape);
    auto dim_mean = vec_to_tuple_ptr({0});
    auto keepdim = std::make_shared<BoolImm>(false);
    auto out_dtype = std::make_shared<Int64Imm>(running_mean_.value()->data_type());
    auto mean_mean = mean_ext(running_mean_.value(), dim_mean, keepdim, out_dtype);
    auto var_mean = mean_ext(running_var_.value(), dim_mean, keepdim, out_dtype);
    inplace_copy(running_mean.value(), mean_mean);
    inplace_copy(running_var.value(), var_mean);
  }

  auto ori_input_shape_tuple_ptr = vec_to_tuple_ptr(input_shape_ori);
  auto out = view(out_bn, ori_input_shape_tuple_ptr);
  op->set_outputs({out});
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
