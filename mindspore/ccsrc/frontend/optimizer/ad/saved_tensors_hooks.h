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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_SAVED_TENSORS_HOOKS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_SAVED_TENSORS_HOOKS_H_

#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace ad {

class WithSavedTensorsHooks {
 public:
  explicit WithSavedTensorsHooks(const FuncGraphPtr &func_graph);
  ~WithSavedTensorsHooks();

 private:
  bool has_saved_tensors_hooks_{false};
};

bool ApplySavedTensorsHooksOnK(const FuncGraphPtr &k, const FuncGraphPtr &bprop_env_fg,
                               const FuncGraphPtr currrent_primal_fg, const CNodePtr &cnode,
                               const AnfNodePtr &out_value, const FuncGraphManagerPtr &manager,
                               const std::vector<AnfNodePtr> &transf_args);
}  // namespace ad
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_SAVED_TENSORS_HOOKS_H_
