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

#include "plugin/res_manager/ascend/op_adapter/op_declare/quantize_ops_declare.h"
#include <string>
#include <vector>

namespace mindspore::device::ascend {
// AscendQuant
INPUT_MAP(AscendQuant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AscendQuant) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                         {"offset", ATTR_DESC(offset, AnyTraits<float>())},
                         {"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                         {"round_mode", ATTR_DESC(round_mode, AnyTraits<std::string>())},
                         {"dst_type", ATTR_DESC(dst_type, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(AscendQuant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendQuant, kNameAscendQuant, ADPT_DESC(AscendQuant))

// AscendQuantV2
INPUT_MAP(AscendQuantV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(scale)}, {3, INPUT_DESC(offset)}};
ATTR_MAP(AscendQuantV2) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(AscendQuantV2) = {{4, ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                                 {5, ATTR_DESC(round_mode, AnyTraits<AscendQuantRoundMode>())},
                                 {6, ATTR_DESC(dst_type, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(AscendQuantV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendQuantV2, kNameAscendQuantV2, ADPT_DESC(AscendQuantV2))

// AscendDequant
INPUT_MAP(AscendDequant) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(deq_scale)}};
ATTR_MAP(AscendDequant) = {{"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                           {"relu_flag", ATTR_DESC(relu_flag, AnyTraits<bool>())},
                           {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(AscendDequant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendDequant, kNameAscendDequant, ADPT_DESC(AscendDequant))

INPUT_MAP(AscendAntiQuant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AscendAntiQuant) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                             {"offset", ATTR_DESC(offset, AnyTraits<float>())},
                             {"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                             {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(AscendAntiQuant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendAntiQuant, kNameAscendAntiQuant, ADPT_DESC(AscendAntiQuant))

// QuantBatchMatmulV3
INPUT_MAP(QuantBatchMatmulV3) = {{1, INPUT_DESC(x1)},     {2, INPUT_DESC(x2)},   {3, INPUT_DESC(scale)},
                                 {4, INPUT_DESC(offset)}, {5, INPUT_DESC(bias)}, {6, INPUT_DESC(pertoken_scale)}};
ATTR_MAP(QuantBatchMatmulV3) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(QuantBatchMatmulV3) = {{7, ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                                      {8, ATTR_DESC(transpose_x2, AnyTraits<bool>())},
                                      {9, ATTR_DESC(dtype, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(QuantBatchMatmulV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(QuantBatchMatmul, kNameQuantBatchMatmul, ADPT_DESC(QuantBatchMatmulV3))
REG_ADPT_DESC(QuantBatchMatmulV3, kNameQuantBatchMatmulV3, ADPT_DESC(QuantBatchMatmulV3))

// QuantConv2D
INPUT_MAP(QuantConv2D) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(scale)}, {4, INPUT_DESC(bias)}, {5, INPUT_DESC(offset)}};
ATTR_MAP(QuantConv2D) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>(), AnyTraits<int64_t>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"pads", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"dilations", ATTR_DESC(dilations, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
                         {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                         {"offset_x", ATTR_DESC(offset_x, AnyTraits<int64_t>())},
                         {"round_mode", ATTR_DESC(round_mode, AnyTraits<std::string>())}};
OUTPUT_MAP(QuantConv2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(QuantConv2D, kNameQuantConv2D, ADPT_DESC(QuantConv2D))

// AscendAntiQuantV2
INPUT_MAP(AscendAntiQuantV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(scale)}, {3, INPUT_DESC(offset)}};
ATTR_MAP(AscendAntiQuantV2) = {{"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                               {"dtype", ATTR_DESC(dst_type, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(AscendAntiQuantV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendAntiQuantV2, kNameAscendAntiQuantV2, ADPT_DESC(AscendAntiQuantV2))

// WeightQuantBatchMatmulV2
INPUT_MAP(WeightQuantBatchMatmulV2) = {{1, INPUT_DESC(x)},
                                       {2, INPUT_DESC(weight)},
                                       {3, INPUT_DESC(antiquant_scale)},
                                       {4, INPUT_DESC(antiquant_offset)},
                                       {5, INPUT_DESC(quant_scale)},
                                       {6, INPUT_DESC(quant_offset)},
                                       {7, INPUT_DESC(bias)}};
ATTR_MAP(WeightQuantBatchMatmulV2) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(WeightQuantBatchMatmulV2) = {{8, ATTR_DESC(transpose_x, AnyTraits<bool>())},
                                            {9, ATTR_DESC(transpose_weight, AnyTraits<bool>())},
                                            {10, ATTR_DESC(antiquant_group_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(WeightQuantBatchMatmulV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(WeightQuantBatchMatmulV2, kNameWeightQuantBatchMatmul, ADPT_DESC(WeightQuantBatchMatmulV2))

// DynamicQuant
INPUT_MAP(DynamicQuant) = {
  {1, INPUT_DESC(x)},
  {2, INPUT_DESC(smooth_scales)},
};
ATTR_MAP(DynamicQuant) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DynamicQuant) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(scale)}};
REG_ADPT_DESC(DynamicQuant, kNameDynamicQuantExt, ADPT_DESC(DynamicQuant))
}  // namespace mindspore::device::ascend
