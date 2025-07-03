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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_SUMMARY_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_SUMMARY_H_

#include <tuple>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "backend/ge_backend/graph_ir/types.h"
#include "ge/ge_graph_compile_summary.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
struct GraphSummary {
  size_t const_memory_size = 0;
  size_t fixed_memory_size = 0;
  size_t workspace_memory_size = 0;
  bool is_refreshable = false;
  size_t stream_num = 0;
  size_t event_num = 0;
  std::vector<ShapeVector> output_shapes = {};
  std::vector<ge::DataType> output_dtypes = {};
  // pair<input_index, output_index>
  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  bool is_static = false;

  GraphSummary() = default;
  explicit GraphSummary(const ::ge::CompiledGraphSummaryPtr &graph_summary);
  std::string ToString() const;

 private:
  std::string TransGeDtypeToString(const backend::ge_backend::GeDataType dtype) const;
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_SUMMARY_H_
