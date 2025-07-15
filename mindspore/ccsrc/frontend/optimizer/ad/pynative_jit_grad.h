/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "pynative/base.h"
#include "mindspore/core/include/ir/manager.h"

namespace mindspore {
namespace ad {
constexpr auto kTopCellWithRecompute = "top_cell_with_recompute";
constexpr auto kOutputNoRecompute = "output_no_recompute";

class BpropGenerator {
 public:
  BpropGenerator(const FuncGraphPtr &fprop_graph, const abstract::AbstractBasePtrList &input_abs,
                 const std::vector<ValuePtr> &input_value, const abstract::AbstractBasePtr &out_abs, bool need_reuse)
      : fprop_graph_(fprop_graph),
        input_abs_(input_abs),
        input_value_(input_value),
        out_abs_(out_abs),
        need_reuse_forward_node_(need_reuse) {
    Init();
  }
  ~BpropGenerator() = default;

  void Init();
  FuncGraphPtr GenerateBpropGraph();
  FuncGraphPtr GenerateForwardGraph(const FuncGraphPtr &jit_forward_graph, bool do_renormalize);
  void SetForwardOutputAbs(const abstract::AbstractBasePtr &forward_abs, const FuncGraphPtr &bprop_graph);
  void EraseUnusedReuseCNode(const FuncGraphPtr &bprop_fg);

 private:
  void ReusePrimalCNode(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg, bool top_cell_do_recompute);
  void ReuseCustomBpropForwardOutput(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg);

  FuncGraphPtr fprop_graph_;
  FuncGraphPtr forward_graph_;
  FuncGraphPtr basic_graph_;
  abstract::AbstractBasePtrList input_abs_{};
  std::vector<ValuePtr> input_value_{};
  abstract::AbstractBasePtr out_abs_{nullptr};
  bool need_reuse_forward_node_{false};
  size_t bprop_origin_param_size_{0};
  std::vector<FuncGraphPtr> fprop_sub_fgs_{};
  AnfNodePtrList replace_nodes_{};
  abstract::AbstractBasePtrList replace_nodes_abs_{};
};
using BpropGeneratorPtr = std::shared_ptr<BpropGenerator>;

FRONTEND_EXPORT std::pair<bool, FuncGraphPtr> GetBpropGraph(const pynative::GradParamPtr &grad_param);
FRONTEND_EXPORT void CheckBpropGraphHasInvalidDout(const std::string &cache_key, const std::vector<bool> &need_grads);
FRONTEND_EXPORT void ClearGradCache();
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H
