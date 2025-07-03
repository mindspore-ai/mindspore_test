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

#include "kernel/ascend/pyboost/customize/argsort.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/ascend/pyboost/auto_generate/sort_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ArgSortAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x,
                                         const Int64ImmPtr &dim, const BoolImmPtr &descending,
                                         const BoolImmPtr &stable) {
  MS_LOG(DEBUG) << "ArgSort call start";
  const auto sort_op = CREATE_PYBOOST_OP(SortExt, op->device_context()->device_context_key().device_name_);
  sort_op->Call(input_x, dim, descending, stable);
  auto indices = sort_op->output(kIndex1);
  op->set_outputs({indices});

  return indices;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
