/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/pyboost/functions/customize/view_impl.h"

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <memory>

#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"
#include "mindspore/core/include/utils/stream_guard.h"
#include "mindspore/ops/view/reshape_strides_calc.h"
#include "mindspore/ccsrc/pyboost/auto_generate/view.h"
#include "mindspore/ccsrc/pyboost/auto_generate/imag_view.h"
#include "mindspore/ccsrc/pyboost/auto_generate/real_view.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_reg.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"

namespace mindspore::kernel::pyboost {
namespace {
inline device::DeviceType GetDeviceTarget() { return OpRunStatus::Get().device_target(); }
}  // namespace

mindspore::tensor::TensorPtr reshape_impl(const mindspore::tensor::TensorPtr &input,
                                          const std::vector<int64_t> &shape) {
  static auto reshape_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().ReshapeGradFuncObj;

  auto storage_info = ops::ReshapeBasicTypeCalc(input, shape);
  const auto &device_target = GetDeviceTarget();
  if (MS_LIKELY(storage_info)) {
    OpRunStatus::Get().HeterBarrier(device_target);
    MS_LOG(DEBUG) << "View contiguous Reshape Call start";
    tensor::TensorPtrList outputs;
    // device info
    const auto &device_context = runtime::OpRunner::GetDeviceContext(device_target);
    auto cur_stream_id = CurrentStream::id();

    kernel::pyboost::PyBoostUtils::PrepareOpInputs(device_context, cur_stream_id, input);
    kernel::pyboost::PyBoostUtils::CreateOutputTensor(device_context, input, storage_info, &outputs);

    // Async
    kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([input, device_context]() {
      MS_LOG(DEBUG) << "View device task Reshape start";
      kernel::pyboost::PyBoostUtils::MallocOpInputsForView(device_context, input);
      MS_LOG(DEBUG) << "View device task Reshape end";
    }));
    reshape_grad_func(outputs[0], input, shape);
    MS_LOG(DEBUG) << "View contiguous Reshape Call end";
    return outputs[0];
  }

  const auto contig_tensor = contiguous(input);
  const auto view_op = CREATE_PYBOOST_OP(View, device_target);
  auto output = view_op->Call(contig_tensor, shape);
  IsSafeViewGuard safe_view_guard(false);
  reshape_grad_func(output, contig_tensor, shape);
  return output;
}

mindspore::tensor::TensorPtr real_view_impl(const mindspore::tensor::TensorPtr &input) {
  const auto &device_target = GetDeviceTarget();
  static auto real_view_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().RealViewGradFuncObj;

  const auto view_op = CREATE_PYBOOST_OP(RealView, device_target);
  auto output = view_op->Call(input);
  real_view_grad_func(output, input);
  return output;
}

mindspore::tensor::TensorPtr imag_view_impl(const mindspore::tensor::TensorPtr &input) {
  const auto &device_target = GetDeviceTarget();
  static auto imag_view_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().ImagViewGradFuncObj;

  const auto view_op = CREATE_PYBOOST_OP(ImagView, device_target);
  auto output = view_op->Call(input);
  imag_view_grad_func(output, input);
  return output;
}

mindspore::tensor::TensorPtr flatten_ext_impl(const mindspore::tensor::TensorPtr &input, const int64_t &start_dim,
                                              const int64_t &end_dim) {
  const auto &input_shape = input->shape();
  const int64_t ndim = input_shape.size();
  auto start = ops::DynamicDimWrap(start_dim, ndim, true);
  auto end = ops::DynamicDimWrap(end_dim, ndim, true);
  if (MS_UNLIKELY(start > end)) {
    MS_EXCEPTION(ValueError) << "For 'flatten', 'start_dim' cannot come after 'end_dim'.";
  }

  if (ndim == 0) {
    return reshape_impl(input, {1});
  }
  if (start == end) {
    return input;
  }

  int64_t slice_numel =
    std::accumulate(input_shape.begin() + start, input_shape.begin() + end + 1, int64_t(1), std::multiplies<int64_t>());
  std::vector<int64_t> out_shape;
  out_shape.reserve(ndim - end + start);
  for (int64_t i = 0; i < start; i++) {
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(slice_numel);
  for (int64_t i = end + 1; i < ndim; i++) {
    out_shape.push_back(input_shape[i]);
  }
  return reshape_impl(input, out_shape);
}

mindspore::tensor::TensorPtr t_ext_impl(const mindspore::tensor::TensorPtr &input) {
  const auto input_rank = input->shape().size();
  return transpose_ext_view(input, 0, input_rank < 2 ? 0 : 1);
}

mindspore::tensor::TensorPtr view_as_impl(const mindspore::tensor::TensorPtr &input,
                                          const mindspore::tensor::TensorPtr &other) {
  return reshape_impl(input, other->shape());
}

mindspore::tensor::TensorPtr expand_as_impl(const mindspore::tensor::TensorPtr &input,
                                            const mindspore::tensor::TensorPtr &other) {
  return broadcast_to(input, other->shape());
}
}  // namespace mindspore::kernel::pyboost
