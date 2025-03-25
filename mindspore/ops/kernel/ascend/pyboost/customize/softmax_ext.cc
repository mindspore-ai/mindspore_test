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

#include <string>
#include "kernel/ascend/pyboost/customize/softmax_ext.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void SoftmaxExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                               const std::optional<Int64ImmPtr> &dim, const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "SoftmaxExt Launch start";
  BaseTensorPtr softmax_out = nullptr;
  ValueTuplePtr axis = nullptr;

  if (!dim.has_value()) {
    MS_LOG(EXCEPTION) << "In SoftmaxExt, dim should not be None.";
  }
  axis = std::make_shared<ValueTuple>(std::vector<ValuePtr>({dim.value()}));

  if (dtype.has_value()) {
    auto converted_dtype = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));
    auto converted_tensor = cast(input_tensor, std::make_shared<Int64Imm>(converted_dtype));
    softmax_out = softmax(converted_tensor, axis);
  } else {
    softmax_out = softmax(input_tensor, axis);
  }
  op->set_outputs({softmax_out});
  MS_LOG(DEBUG) << "SoftmaxExt Launch end";

  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
