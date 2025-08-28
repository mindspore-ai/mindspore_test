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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_POSTPROCESSOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_POSTPROCESSOR_H_

#include <vector>
#include "frontend/parallel/parallel_processor_context.h"

namespace mindspore {
namespace parallel {
class ParallelPostprocessor {
 public:
  explicit ParallelPostprocessor(const ParallelProcessorContextPtr &context) : processor_context_(context) {
    MS_EXCEPTION_IF_NULL(processor_context_);
  }

  void Process();

 private:
  void PipelinePostProcessStep1();
  void PipelinePostProcessStep2();
  const ParallelProcessorContextPtr &processor_context_;
};

void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, StrategyMap *stra_map,
                        TensorInfoMap *tensor_info_map, ManualShapeMap *manual_shape_map);
void CheckpointOnline(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);
void CheckpointOffline(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_POSTPROCESSOR_H_
