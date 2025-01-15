/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/pyboost/customize/reshape_and_cache.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeAndCacheAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &key,
                                                     const BaseTensorPtr &value, const BaseTensorPtr &key_cache,
                                                     const BaseTensorPtr &value_cache,
                                                     const BaseTensorPtr &slot_mapping) {
  OpRunner::InferOpOutput(op, key, value, key_cache, value_cache, slot_mapping);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), key, value, key_cache, value_cache,
                                slot_mapping);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  InternalAscendCall(op, key, value, key_cache, value_cache, slot_mapping);
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
