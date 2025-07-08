/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include "utils/log_adapter.h"
#include "graph/operator.h"

namespace mindspore::device::ascend {
namespace {
bool InferOffline(const ge::Operator &op, std::vector<ge::TensorDesc> *outputs_info) {
  if (outputs_info == nullptr) {
    return false;
  }

  // output_shapes
  std::vector<std::vector<int64_t>> output_shapes;
  if (op.GetAttr("output_shapes", output_shapes) != ge::GRAPH_SUCCESS) {
    return false;
  }

  // output_formats
  std::vector<int32_t> output_formats;
  if (op.GetAttr("output_formats", output_formats) != ge::GRAPH_SUCCESS ||
      output_formats.size() != output_shapes.size()) {
    return false;
  }

  // output_types
  std::vector<int32_t> output_types;
  if (op.GetAttr("output_types", output_types) != ge::GRAPH_SUCCESS || output_types.size() != output_shapes.size()) {
    return false;
  }

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    (void)outputs_info->emplace_back(ge::Shape(output_shapes[i]), static_cast<ge::Format>(output_formats[i]),
                                     static_cast<ge::DataType>(output_types[i]));
  }
  return true;
}

std::string GetCustomOpName(const ge::Operator &op) {
  std::string res;
  ge::AscendString op_name;
  if (op.GetName(op_name) != ge::GRAPH_SUCCESS) {
    return res;
  }
  return op_name.GetString();
}

std::string GetCustomOpType(const ge::Operator &op) {
  std::string res;
  ge::AscendString op_type;
  if (op.GetOpType(op_type) != ge::GRAPH_SUCCESS) {
    return res;
  }
  return op_type.GetString();
}

std::string GetCustomOpKey(const ge::Operator &op) {
  auto op_name = GetCustomOpName(op);
  auto op_type = GetCustomOpType(op);
  auto op_key = op_name + "(" + op_type + ")";
  return op_key;
}
}  // namespace

ge::graphStatus CustomAkgOpInferFunc(ge::Operator &) { return ge::GRAPH_SUCCESS; }

ge::graphStatus CustomTbeAicpuOpInferFunc(ge::Operator &op) {
  auto op_key = GetCustomOpKey(op);
  MS_LOG(INFO) << "Start infer shape for op " << op_key;
  std::vector<ge::TensorDesc> outputs_info;
  if (!InferOffline(op, &outputs_info)) {
    MS_LOG(ERROR) << "Failed infer shape for op " << op_key;
    return ge::GRAPH_FAILED;
  }
  // update output desc
  std::vector<std::string> output_names;
  if (op.GetAttr("output_names", output_names) != ge::GRAPH_SUCCESS || output_names.size() != outputs_info.size()) {
    MS_LOG(ERROR) << "For op " << op_key
                  << ", attr 'output_names' size is not equal to outputs_info size: " << output_names.size() << " vs "
                  << outputs_info.size();
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    (void)op.UpdateOutputDesc(output_names[i], outputs_info[i]);
  }
  MS_LOG(INFO) << "End infer shape for op " << op_key;
  return ge::GRAPH_SUCCESS;
}
}  // namespace mindspore::device::ascend
