/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/op_adapter/io_format_map.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_util.h"
#include "utils/check_convert_utils.h"

namespace mindspore::device::ascend {
mindspore::HashMap<std::string, std::string> IOFormatMap::io_format_map_ = {{"BNTrainingReduce", "NCHW"},
                                                                            {"BNTrainingUpdate", "NCHW"},
                                                                            {"BNTrainingUpdateGrad", "NCHW"},
                                                                            {"BNTrainingReduceGrad", "NCHW"},
                                                                            {"BNInfer", "NCHW"},
                                                                            {"FusedBatchNorm", "NCHW"},
                                                                            {"BNInferGrad", "NCHW"},
                                                                            {"Conv2D", "NCHW"},
                                                                            {"Transpose", "ND"},
                                                                            {"DepthwiseConv2D", "NCHW"},
                                                                            {"DepthwiseConv2dNative", "NCHW"},
                                                                            {"Conv2DBackpropInput", "NCHW"},
                                                                            {"Conv2DBackpropFilter", "NCHW"},
                                                                            {"BasicLSTMCellWeightGrad", "HWCN"},
                                                                            {"ExtractImagePatches", "NCHW"},
                                                                            {"FullConnection", "NCHW"},
                                                                            {"PReLU", "NCHW"},
                                                                            {"Scale", "NCHW"},
                                                                            {"GridSampler2D", "NCHW"},
                                                                            {"ResizeBilinearV2", "NCHW"},
                                                                            {"ResizeNearestNeighborV2", "NCHW"},
                                                                            {"Conv3D", "format"},
                                                                            {"MaxPool3D", "NCDHW"},
                                                                            {"MaxPoolV3", "NCHW"},
                                                                            {"MaxPool3DGrad", "NCDHW"},
                                                                            {"AvgPool3D", "NCDHW"},
                                                                            {"AvgPool3DGrad", "NCDHW"},
                                                                            {"Conv3DBackpropFilter", "format"},
                                                                            {"Conv3DBackpropInput", "format"},
                                                                            {"Conv3DTranspose", "format"},
                                                                            {"DepthToSpace", "format"},
                                                                            {"DeformableOffsetsGrad", "format"},
                                                                            {"ExtractVolumePatches", "format"},
                                                                            {"DeformableConv2d", "NCHW"},
                                                                            {"Conv2DTransposeV2", "NCHW"},
                                                                            {"Col2Im", "NCHW"},
                                                                            {"SpaceToDepth", "NCHW"},
                                                                            {"Pooling", "NCHW"},
                                                                            {"AvgPoolV2", "NCHW"},
                                                                            {"QuantConv2D", "NCHW"},
                                                                            {"GridSampler3D", "NCDHW"},
                                                                            {"ResizeD", "NCHW"}};
mindspore::HashMap<std::string, std::string> &IOFormatMap::get() { return io_format_map_; }

std::string GetOpIOFormat(const AnfNodePtr &anf) {
  std::string ret;
  if (anf == nullptr) {
    MS_LOG(ERROR) << "The anf is nullptr";
    return ret;
  }
  auto node = anf->cast<CNodePtr>();
  if (node == nullptr) {
    MS_LOG(ERROR) << "The anf is not a cnode.";
    return ret;
  }
  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Length of node inputs is empty.";
  }
  MS_EXCEPTION_IF_NULL(node->input(0));
  auto &input = node->input(0);
  AnfNodePtr prim_node = nullptr;
  if (input->isa<ValueNode>()) {
    prim_node = input;
  } else if (input->isa<CNode>() && input->cast<CNodePtr>()->input(0)->isa<ValueNode>()) {
    // process cnode1, its input(index 0) is a conde0(partial etc.)
    prim_node = input->cast<CNodePtr>()->input(0);
  } else {
    MS_LOG(ERROR) << "The anf is not a value node or cnode.";
    return ret;
  }
  MS_EXCEPTION_IF_NULL(prim_node);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "The anf is not a Primitive.";
    return ret;
  }
  if (prim->HasAttr("io_format")) {
    return GetValueWithCheck<std::string>(prim->GetAttr("io_format"));
  }
  auto io_format_map = IOFormatMap::get();
  auto iter = io_format_map.find(prim->name());
  if (iter == io_format_map.end()) {
    return kOpFormat_DEFAULT;
  }
  if (iter->second == "format") {
    ValuePtr format = prim->GetAttr("format");
    MS_EXCEPTION_IF_NULL(format);
    if (format->isa<Int64Imm>()) {
      bool converted = CheckAndConvertUtils::ConvertAttrValueToString(prim->name(), "format", &format);
      if (converted) {
        return GetValueWithCheck<std::string>(format);
      }
    } else {
      return GetValueWithCheck<std::string>(format);
    }
  }
  return iter->second;
}
}  // namespace mindspore::device::ascend
