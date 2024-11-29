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

#include "runtime/pipeline/task/run_graph_task.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void RunGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(func_);
  func_();
}

void RunGraphTask::SetException(const std::exception_ptr &e) {
  if (stub_output_) {
    stub_output_->SetException(e);
  }
}
}  // namespace runtime
}  // namespace mindspore
