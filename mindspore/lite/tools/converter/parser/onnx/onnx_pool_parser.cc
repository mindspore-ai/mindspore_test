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

#include "tools/converter/parser/onnx/onnx_pool_parser.h"
#include <memory>
#include <vector>
#include "infer/cxx_api/avg_pool_fusion.h"
#include "infer/cxx_api/max_pool_fusion.h"
#include "include/registry/converter_context.h"
#include "nnacl/op_base.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNumShapeSize2 = 2;
constexpr size_t kNumShapeSize3 = 3;
constexpr size_t kNumShapeSize4 = 4;
constexpr size_t kNumShapeSize6 = 6;
bool CheckDilations(const onnx::AttributeProto &onnx_node_attr) {
  for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
    if (onnx_node_attr.ints(i) != 1) {
      MS_LOG(ERROR) << "Pool op only support dilations=<1, 1> now!";
      return false;
    }
  }
  return true;
}

bool ParseAttrs(const onnx::NodeProto &onnx_node, const std::unique_ptr<ops::AvgPoolFusion> &prim,
                std::vector<int64_t> *kernels, std::vector<int64_t> *strides, std::vector<int64_t> *pads,
                mindspore::RoundMode *round_mode, bool *is_3d) {
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == kNumShapeSize2) {
        kernels->push_back(onnx_node_attr.ints(0));
        kernels->push_back(onnx_node_attr.ints(kIndex1));
        prim->set_kernel_size(*kernels);
      } else if (onnx_node_attr.ints_size() == kNumShapeSize3) {
        *is_3d = true;
        kernels->push_back(onnx_node_attr.ints(0));
        kernels->push_back(onnx_node_attr.ints(kIndex1));
        kernels->push_back(onnx_node_attr.ints(kIndex2));
        prim->AddAttr("kernel_size", api::MakeValue<std::vector<int64_t>>(*kernels));
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == kNumShapeSize2) {
        strides->push_back(onnx_node_attr.ints(0));
        strides->push_back(onnx_node_attr.ints(kIndex1));
      } else if (onnx_node_attr.ints_size() == kNumShapeSize3) {
        *is_3d = true;
        strides->push_back(onnx_node_attr.ints(0));
        strides->push_back(onnx_node_attr.ints(kIndex1));
        strides->push_back(onnx_node_attr.ints(kIndex2));
      }
    }
    if (attribute_name == "auto_pad") {
      if (onnx_node_attr.s() == "SAME_UPPER") {
        prim->set_pad_mode(mindspore::PadMode::SAME);
      } else if (onnx_node_attr.s() == "SAME_LOWER") {
        MS_LOG(ERROR) << "PadMode_SAME_LOWER is not supported now";
        return false;
      }
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == kNumShapeSize4) {
        pads->push_back(onnx_node_attr.ints(0));
        pads->push_back(onnx_node_attr.ints(kIndex2));
        pads->push_back(onnx_node_attr.ints(kIndex1));
        pads->push_back(onnx_node_attr.ints(kIndex3));
      } else if (onnx_node_attr.ints_size() == kNumShapeSize6) {
        *is_3d = true;
        pads->push_back(onnx_node_attr.ints(0));
        pads->push_back(onnx_node_attr.ints(kIndex3));
        pads->push_back(onnx_node_attr.ints(kIndex1));
        pads->push_back(onnx_node_attr.ints(kIndex4));
        pads->push_back(onnx_node_attr.ints(kIndex2));
        pads->push_back(onnx_node_attr.ints(kIndex5));
      }
    }
    if (attribute_name == "ceil_mode") {
      *round_mode = (onnx_node_attr.i() == 0) ? mindspore::RoundMode::FLOOR : mindspore::RoundMode::CEIL;
    }
    if (attribute_name == ops::kCountIncludePad) {
      bool include = onnx_node_attr.i() == 0 ? false : true;
      (void)prim->AddAttr(ops::kCountIncludePad, api::MakeValue(include));
    }
    MS_CHECK_TRUE_MSG(!(attribute_name == "dilations" && !CheckDilations(onnx_node_attr)), false,
                      "dilations is invalid!");
  }
  return true;
}

}  // namespace

PrimitiveCPtr OnnxAvgPoolParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::AvgPoolFusion>();
  bool is_3d = false;
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));
  prim->set_pad_mode(mindspore::PadMode::PAD);
  mindspore::RoundMode round_mode = mindspore::RoundMode::FLOOR;
  std::vector<int64_t> kernels;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  if (!ParseAttrs(onnx_node, prim, &kernels, &strides, &pads, &round_mode, &is_3d)) {
    MS_LOG(ERROR) << "ParseAttrs failed!";
    return nullptr;
  }
  prim->set_round_mode(round_mode);

  if (strides.empty()) {
    strides.push_back(1);
    strides.push_back(1);
  }
  prim->set_strides(strides);
  if (pads.empty()) {
    pads = is_3d ? std::vector<int64_t>(kNumShapeSize6, 0) : std::vector<int64_t>(kNumShapeSize4, 0);
  }
  prim->set_pad(pads);
  prim->set_global(onnx_node.op_type() == "GlobalAveragePool");
  int fmk_type = converter::FmkType::kFmkTypeOnnx;
  (void)prim_c->AddAttr(ops::kFmkType, MakeValue(fmk_type));
  return prim->GetPrim();
}

PrimitiveCPtr OnnxMaxPoolParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::MaxPoolFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));
  prim->set_kernel_size({1, 1});
  mindspore::RoundMode round_mode = mindspore::RoundMode::FLOOR;
  std::vector<int64_t> kernels;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == kNumShapeSize2) {
        kernels.push_back(onnx_node_attr.ints(0));
        kernels.push_back(onnx_node_attr.ints(1));
        prim->set_kernel_size(kernels);
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == kNumShapeSize2) {
        strides.push_back(onnx_node_attr.ints(0));
        strides.push_back(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "auto_pad") {
      if (onnx_node_attr.s() == "SAME_UPPER") {
        prim->set_pad_mode(mindspore::PadMode::SAME);
      } else if (onnx_node_attr.s() == "SAME_LOWER") {
        MS_LOG(ERROR) << "PadMode_SAME_LOWER is not supported now";
        return nullptr;
      }
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == kNumShapeSize4) {
        prim->set_pad_mode(mindspore::PadMode::PAD);
        pads.push_back(onnx_node_attr.ints(0));
        pads.push_back(onnx_node_attr.ints(2));
        pads.push_back(onnx_node_attr.ints(1));
        pads.push_back(onnx_node_attr.ints(3));
      }
    }
    if (attribute_name == "ceil_mode") {
      if (onnx_node_attr.i() == 0) {
        round_mode = mindspore::RoundMode::FLOOR;
      } else {
        round_mode = mindspore::RoundMode::CEIL;
      }
    }
    if (attribute_name == "dilations") {
      if (onnx_node_attr.ints_size() != kNumShapeSize2 || onnx_node_attr.ints(0) != 1 || onnx_node_attr.ints(1) != 1) {
        MS_LOG(ERROR) << "MaxPool op only support dilations=<1, 1> now!";
        return nullptr;
      }
    }
  }
  prim->set_round_mode(round_mode);

  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  prim->set_pad(pads);

  if (strides.empty()) {
    strides.push_back(1);
    strides.push_back(1);
  }
  prim->set_strides(strides);
  prim->set_global(onnx_node.op_type() == "GlobalMaxPool");
  int fmk_type = converter::FmkType::kFmkTypeOnnx;
  (void)prim_c->AddAttr(ops::kFmkType, MakeValue(fmk_type));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxAveragePoolParser("AveragePool", new OnnxAvgPoolParser());
OnnxNodeRegistrar g_onnxGlobalAveragePoolParser("GlobalAveragePool", new OnnxAvgPoolParser());
OnnxNodeRegistrar g_onnxInt8AveragePoolParser("Int8AveragePool", new OnnxAvgPoolParser());

OnnxNodeRegistrar g_onnxMaxPoolParser("MaxPool", new OnnxMaxPoolParser());
OnnxNodeRegistrar g_onnxGlobalMaxPoolParser("GlobalMaxPool", new OnnxMaxPoolParser());
}  // namespace lite
}  // namespace mindspore
