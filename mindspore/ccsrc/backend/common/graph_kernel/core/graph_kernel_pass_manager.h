/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_PASS_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_PASS_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/optimizer/pass_manager.h"

namespace mindspore::graphkernel {
using opt::PassManager;

class BACKEND_COMMON_EXPORT GraphKernelPassManager : public PassManager {
 public:
  GraphKernelPassManager(size_t stage, const std::string &name) : PassManager(name, true), stage_(stage) {}
  ~GraphKernelPassManager() = default;

  // Add graph pass, the pass object will be freed when pass manager freed.
  void Add(const opt::PassPtr &pass, unsigned int pass_level = OptLevel_0, bool supported_device = true);

  // Run passes on the func_graph
  bool Run(const FuncGraphPtr &func_graph) const override;

 protected:
  bool RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const opt::PassPtr &pass) const override;
  std::string GetPassFullname(size_t pass_id, const opt::PassPtr &pass) const override;

 private:
  size_t stage_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_PASS_MANAGER_H_
