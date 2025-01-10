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

#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

void feed_input_args(GeneralInferParamGenerator &generator, ShapeVector self_shape, TypeId self_type,
                     ShapeVector x2_shape, TypeId x2_type, int64_t world_size, std::optional<bool> gather_output,
                     std::optional<bool> trans_self, std::optional<bool> trans_x2) {
  generator.FeedInputArgs(
    {InferInfoParam{self_shape, self_type}, InferInfoParam{x2_shape, x2_type},
     InferInfoParam{ShapeVector{}, kObjectTypeString, MakeValue("hccl_world_group")},
     InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(world_size)},
     InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
     InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(0)},
     InferInfoParam{ShapeVector{}, kNumberTypeBool,
                    gather_output.has_value() ? MakeValue(gather_output.value()) : kValueAny},
     InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(0)},
     InferInfoParam{ShapeVector{}, kNumberTypeBool, trans_self.has_value() ? MakeValue(trans_self.value()) : kValueAny},
     InferInfoParam{ShapeVector{}, kNumberTypeBool, trans_x2.has_value() ? MakeValue(trans_x2.value()) : kValueAny}});
}

void add_successul_case(GeneralInferParamGenerator &generator, ShapeVector self_shape, TypeId self_type,
                        ShapeVector x2_shape, TypeId x2_type, int64_t world_size, std::optional<bool> gather_output,
                        std::optional<bool> trans_self, std::optional<bool> trans_x2, ShapeVector output_shape,
                        TypeId output_type, ShapeVector gather_out_shape, TypeId gather_out_type) {
  feed_input_args(generator, self_shape, self_type, x2_shape, x2_type, world_size, gather_output, trans_self, trans_x2);
  generator.FeedExpectedOutput({output_shape, gather_out_shape}, {output_type, gather_out_type});
}

void add_failed_case(GeneralInferParamGenerator &generator, ShapeVector self_shape, TypeId self_type,
                     ShapeVector x2_shape, TypeId x2_type, int64_t world_size, std::optional<bool> gather_output,
                     std::optional<bool> trans_self, std::optional<bool> trans_x2) {
  feed_input_args(generator, self_shape, self_type, x2_shape, x2_type, world_size, gather_output, trans_self, trans_x2);
  generator.CaseShouldThrow();
}

void add_successul_shape_case(GeneralInferParamGenerator &generator, ShapeVector self_shape, ShapeVector x2_shape,
                              std::optional<bool> trans_self, std::optional<bool> trans_x2, int64_t world_size,
                              std::optional<bool> gather_output, ShapeVector output_shape,
                              ShapeVector gather_out_shape) {
  add_successul_case(generator, self_shape, kNumberTypeFloat16, x2_shape, kNumberTypeFloat16, world_size, gather_output,
                     trans_self, trans_x2, output_shape, kNumberTypeFloat16, gather_out_shape, kNumberTypeFloat16);
}

void add_failed_shape_case(GeneralInferParamGenerator &generator, ShapeVector self_shape, ShapeVector x2_shape,
                           std::optional<bool> trans_self, std::optional<bool> trans_x2, int64_t world_size,
                           std::optional<bool> gather_output) {
  add_failed_case(generator, self_shape, kNumberTypeFloat16, x2_shape, kNumberTypeFloat16, world_size, gather_output,
                  trans_self, trans_x2);
}

void add_successul_type_case(GeneralInferParamGenerator &generator, TypeId self_type, TypeId x2_type,
                             TypeId output_type, TypeId gather_out_type) {
  add_successul_case(generator, {128, 256}, self_type, {256, 512}, x2_type, 8, true, false, false, {1024, 512},
                     output_type, {1024, 256}, gather_out_type);
}

void add_failed_type_case(GeneralInferParamGenerator &generator, TypeId self_type, TypeId x2_type) {
  add_failed_case(generator, {128, 256}, self_type, {256, 512}, x2_type, 8, true, false, false);
}

void add_bias_not_none_failed_case(GeneralInferParamGenerator &generator) {
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{128, 256}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{256, 512}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kObjectTypeString, MakeValue("hccl_world_group")},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(8)},
                    InferInfoParam{ShapeVector{512}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, MakeValue(true)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, MakeValue<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, MakeValue(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, MakeValue(false)}})
    .CaseShouldThrow();
}

std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  add_successul_shape_case(generator, {128, 256}, {256, 512}, false, false, 8, true, {1024, 512}, {1024, 256});
  add_successul_shape_case(generator, {256, 128}, {256, 512}, true, false, 8, true, {1024, 512}, {1024, 256});
  add_successul_shape_case(generator, {128, 256}, {512, 256}, false, true, 8, true, {1024, 512}, {1024, 256});
  add_successul_shape_case(generator, {128, 256}, {256, 512}, false, false, 8, false, {1024, 512}, {0});
  add_successul_shape_case(generator, {kShapeRankAny}, {256, 512}, false, false, 8, true, {kShapeDimAny, 512},
                           {kShapeDimAny, 256});
  add_successul_shape_case(generator, {128, 256}, {256, 512}, std::nullopt, false, 8, true, {kShapeDimAny, 512},
                           {kShapeDimAny, 256});
  add_successul_shape_case(generator, {kShapeDimAny, 256}, {256, 512}, false, false, 8, true, {kShapeDimAny, 512},
                           {kShapeDimAny, 256});
  add_successul_shape_case(generator, {128, kShapeDimAny}, {256, 512}, false, false, 8, true, {1024, 512}, {1024, 256});
  add_successul_shape_case(generator, {128, kShapeDimAny}, {kShapeDimAny, 512}, false, false, 8, true, {1024, 512},
                           {1024, kShapeDimAny});
  add_successul_shape_case(generator, {128, 256}, {kShapeRankAny}, false, false, 8, true, {1024, kShapeDimAny},
                           {1024, 256});
  add_successul_shape_case(generator, {128, 256}, {256, 512}, false, std::nullopt, 8, true, {1024, kShapeDimAny},
                           {1024, 256});
  add_successul_shape_case(generator, {128, 256}, {256, kShapeDimAny}, false, false, 8, true, {1024, kShapeDimAny},
                           {1024, 256});
  add_successul_shape_case(generator, {128, 256}, {256, 512}, false, false, 8, std::nullopt, {1024, 512},
                           {kShapeRankAny});
  add_failed_shape_case(generator, {128}, {256, 512}, false, false, 8, true);
  add_failed_shape_case(generator, {128, 256}, {256}, false, false, 8, true);
  add_failed_shape_case(generator, {128, 256}, {512, 512}, false, false, 8, true);
  add_successul_type_case(generator, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16);
  add_successul_type_case(generator, kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16,
                          kNumberTypeBFloat16);
  add_failed_type_case(generator, kNumberTypeFloat16, kNumberTypeBFloat16);
  add_bias_not_none_failed_case(generator);
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AllGatherMatmul, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
