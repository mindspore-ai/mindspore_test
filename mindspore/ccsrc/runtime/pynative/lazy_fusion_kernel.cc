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

#include "runtime/pynative/lazy_fusion_kernel.h"
#include "utils/log_adapter.h"

namespace mindspore {
LazyFusionManager g_lazy_fusion_manager;

LazyFusionManager::~LazyFusionManager() {
  while (!pool_.empty()) {
    auto top = pool_.front();
    delete top;
    pool_.pop();
  }
}

void LazyFusionManager::Init() {
  auto build_iter = build_funcs_.find(kAscendDevice);
  if (build_iter != build_funcs_.end()) {
    MS_LOG(INFO) << "Set build_func";
    build_func_ = build_iter->second;
    MS_EXCEPTION_IF_NULL(build_func_);
  }
  auto init_iter = init_funcs_.find(kAscendDevice);
  if (init_iter != init_funcs_.end()) {
    init_func_ = init_iter->second;
    MS_LOG(INFO) << "Set init_func";
    MS_EXCEPTION_IF_NULL(init_func_);
    init_func_();
  }
}
}  // namespace mindspore
