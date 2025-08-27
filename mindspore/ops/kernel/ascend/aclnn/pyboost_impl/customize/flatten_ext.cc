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

#include "kernel/ascend/aclnn/pyboost_impl/customize/flatten_ext.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <algorithm>
#include <vector>
#include <functional>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/auto_generate/reshape.h"
#include "mindspore/ops/view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void FlattenExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x_tensor,
                               const int64_t &start_dim, const int64_t &end_dim) {
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";

  const auto &input_shape = input_x_tensor->shape();
  const int64_t ndim = input_shape.size();

  auto start = ops::DynamicDimWrap(start_dim, ndim, true);
  auto end = ops::DynamicDimWrap(end_dim, ndim, true);
  if (MS_UNLIKELY(start > end)) {
    MS_EXCEPTION(ValueError) << "For 'flatten', 'start_dim' cannot come after 'end_dim'.";
  }

  std::vector<int64_t> out_shape;
  if (ndim == 0) {
    out_shape = std::vector<int64_t>{1};
  } else if (start == end) {
    out_shape = input_shape;
  } else {
    out_shape.reserve(ndim - end + start);
    (void)std::transform(input_shape.begin(), input_shape.begin() + start, std::back_inserter(out_shape),
                         [](int64_t v) { return v; });
    out_shape.push_back(-1);
    (void)std::transform(input_shape.begin() + end + 1, input_shape.end(), std::back_inserter(out_shape),
                         [](int64_t v) { return v; });
  }
  auto reshape_op = CREATE_PYBOOST_OP(Reshape, device::DeviceType::kAscend);
  reshape_op->Call(input_x_tensor, out_shape);
  op->set_outputs(reshape_op->outputs());
  MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
