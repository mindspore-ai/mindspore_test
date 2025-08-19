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

#include "backend/ge_backend/executor/ge_summary.h"
#include <vector>
#include <string>
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
GraphSummary::GraphSummary(const ::ge::CompiledGraphSummaryPtr &graph_summary) {
  MS_EXCEPTION_IF_NULL(graph_summary);
  is_static = graph_summary->IsStatic();
  if (is_static) {
    ::ge::graphStatus status = graph_summary->GetConstMemorySize(const_memory_size);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetConstMemorySize failed, status = " << status;
    }
    status = graph_summary->GetFeatureMemorySize(fixed_memory_size);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetFeatureMemorySize failed, status = " << status;
    }
    status = graph_summary->GetFeatureMemoryBaseRefreshable(is_refreshable);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetFeatureMemoryBaseRefreshable failed, status = " << status;
    }
    if (is_refreshable) {
      status = graph_summary->GetFixedFeatureMemorySize(fixed_memory_size);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetFixedFeatureMemorySize failed, status = " << status;
      }
      status = graph_summary->GetRefreshableFeatureMemorySize(workspace_memory_size);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetRefreshableFeatureMemorySize failed, status = " << status;
      }
    }
    status = graph_summary->GetStreamNum(stream_num);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetStreamNum failed, status = " << status;
    }
    status = graph_summary->GetEventNum(event_num);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetEventNum failed, status = " << status;
    }
    std::vector<::ge::Shape> ge_shapes;
    status = graph_summary->GetOutputShapes(ge_shapes);
    if (status != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetOutputShapes failed, status = " << status;
    }
    (void)std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(output_shapes),
                         [](const ::ge::Shape &ge_shape) -> ShapeVector { return ge_shape.GetDims(); });
    if (graph_summary->GetOutputDtypes(output_dtypes) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetOutputDtypes failed, status = " << status
                        << ", maybe the execution mode is not as expected.";
    }
    if (graph_summary->GetIOIndexesWithSameAddr(io_indexes) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "GetIOIndexesWithSameAddr failed, status = " << status
                        << ", maybe the execution mode is not as expected.";
    }
  } else {
    MS_LOG(WARNING) << "Graph is not static, maybe the execution mode is not as expected.";
  }
}

std::string GraphSummary::ToString() const {
  std::stringstream ss;
  ss << "const_memory_size[" << const_memory_size << "], fixed_memory_size[" << fixed_memory_size
     << "], workspace_memory_size[" << workspace_memory_size << "], is_refreshable[" << is_refreshable
     << "], stream_num[" << stream_num << "], event_num[" << event_num << "], output size[" << output_shapes.size()
     << "], is_static[" << is_static << "]";
  if (!output_shapes.empty()) {
    if (output_shapes.size() != output_dtypes.size()) {
      MS_LOG(WARNING) << "The output_dtypes size in summary is not equal to output_shapes size.";
    }
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      std::string shape_str = "[";
      std::string dtype_str = "";
      for (size_t j = 0; j < output_shapes[i].size(); ++j) {
        if (j != output_shapes[i].size() - 1) {
          shape_str += std::to_string(output_shapes[i][j]) + ",";
        } else {
          shape_str += std::to_string(output_shapes[i][j]) + "]";
        }
      }

      if (output_shapes[i].empty()) {
        shape_str = "[]";
      }
      if (i < output_dtypes.size()) {
        dtype_str += "[";
        dtype_str += TransGeDtypeToString(output_dtypes[i]);
        dtype_str += "]";
      }
      if (dtype_str.empty()) {
        ss << ", output[" << i << "] shape = " << shape_str;
      } else {
        ss << ", output[" << i << "] shape = " << shape_str << " dtype = " << dtype_str;
      }
    }
  }
  if (!io_indexes.empty()) {
    std::string io_indexes_str = "[";
    for (auto io_index : io_indexes) {
      io_indexes_str += "[" + std::to_string(io_index.first) + "," + std::to_string(io_index.second) + "]";
    }
    io_indexes_str += "]";
    ss << ", io_indexes: " << io_indexes_str;
  }

  return ss.str();
}

std::string GraphSummary::TransGeDtypeToString(const backend::ge_backend::GeDataType dtype) const {
  std::string dtype_str = "";
  if (device::ascend::ge_dtype_str_map.find(dtype) != device::ascend::ge_dtype_str_map.end()) {
    dtype_str = device::ascend::ge_dtype_str_map[dtype];
  }
  return dtype_str;
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
