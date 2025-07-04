/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_AXIS_NORMALIZER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_AXIS_NORMALIZER_H_

#include "ir/func_graph.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
// change Reduce nodes' axis to non-negative value
class AxisNormalizer : public opt::Pass {
 public:
  AxisNormalizer() : Pass("axis_normalizer") {}
  ~AxisNormalizer() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const AnfNodePtr &graph_kernel_node) const;
  // convert the axis value to a shape vector, return true if the axis changed after normalization,
  // return false otherwise
  bool AxisProcess(ValuePtr axis, const size_t rank, ShapeVector *axis_vec) const;
  int64_t NormAxis(int64_t x, size_t rank) const;
  bool IsReduce(const AnfNodePtr &node) const;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_AXIS_NORMALIZER_H_
