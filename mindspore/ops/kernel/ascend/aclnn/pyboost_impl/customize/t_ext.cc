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

#include "kernel/ascend/aclnn/pyboost_impl/customize/t_ext.h"
#include <memory>
#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/transpose.h"
#include "mindspore/ccsrc/pyboost/auto_generate/copy.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void TExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "TExt Launch start";

  auto input_rank = input_tensor->shape().size();
  if (MS_UNLIKELY(input_rank > kIndex2)) {
    MS_EXCEPTION(ValueError) << "For TExt, the input rank should be less equal to 2, but got " << input_rank;
  }
  auto transpose_op = CREATE_PYBOOST_OP(Transpose, device::DeviceType::kAscend);
  std::vector<int64_t> perm;
  perm.reserve(input_rank);
  for (size_t i = 0; i < input_rank; ++i) {
    perm.push_back(static_cast<int64_t>(input_rank - i - 1));
  }
  auto output_tensor = transpose_op->Call(input_tensor, perm);
  op->set_outputs({output_tensor});

  MS_LOG(DEBUG) << "TExt Launch end";
  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
