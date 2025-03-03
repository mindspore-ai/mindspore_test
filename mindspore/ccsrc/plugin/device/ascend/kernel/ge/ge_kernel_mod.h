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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GE_GE_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GE_GE_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include "ops/base_operator.h"
#include "common/kernel.h"
#include "common/ms_factory.h"
#include "include/common/utils/anfalgo.h"
#include "backend/ge_backend/executor/ge_graph_executor.h"

namespace mindspore {
namespace kernel {
class GeKernelMod : public KernelMod {
 public:
  // =========================================New interface==========================================================
  GeKernelMod() {}
  ~GeKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool IsNeedUpdateOutputShapeAndSize() override;

  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override {}

  void set_skip_run(bool flag) { skip_run_ = flag; }

  void set_executor(backend::ge_backend::GeGraphExecutor *executor) { graph_executor_ = executor; }

  void set_graph(const KernelGraphPtr &graph) { graph_ = graph; }

  void set_kernel(const AnfNodePtr &node) { node_ = node; }

  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in GE."; }

  void set_io_indexes(const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) { io_indexes_ = io_indexes; }

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes() const { return io_indexes_; }

  void InitGeMemory(size_t stream_id) const;

 protected:
  bool skip_run_{false};
  backend::ge_backend::GeGraphExecutor *graph_executor_ = nullptr;
  std::vector<std::pair<uint32_t, uint32_t>> io_indexes_;
  KernelGraphPtr graph_;
  AnfNodeWeakPtr node_;
};

using GeKernelModPtr = std::shared_ptr<GeKernelMod>;
using GeKernelModPtrList = std::vector<GeKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GE_GE_KERNEL_MOD_H_
