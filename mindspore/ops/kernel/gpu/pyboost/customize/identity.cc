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

#include "kernel/gpu/pyboost/customize/identity.h"
#include <memory>
#include <utility>
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "mindspore/ccsrc/pyboost/customize/identity.h"
#include "include/common/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr IdentityGPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Identity call start";
  IdentityCustomize(op, x_tensor);
  auto sync = runtime::RuntimeConf::GetInstance()->launch_blocking();
  if (sync && !op->device_context()->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for op Identity.";
  }
  MS_LOG(DEBUG) << "Identity call end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
