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

#include <vector>
#include <stdint.h>
#include "custom_aot_extra.h"
enum TypeId : int {};

extern "C" std::vector<int64_t> aclnnAddCustomInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  std::vector<int64_t> output_shape;
  auto input0_size = ndims[0];
  for (size_t i = 0; i < input0_size; i++) {
    output_shape.push_back(shapes[0][i]);
  }
  return output_shape;
}

extern "C" std::vector<int64_t> MulInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  std::vector<int64_t> output_shape;
  auto input0_size = ndims[0];
  for (size_t i = 0; i < input0_size; i++) {
    output_shape.push_back(shapes[0][i]);
  }
  return output_shape;
}

extern "C" TypeId MulInferType(std::vector<TypeId> type_ids, AotExtra *extra) { return type_ids[0]; }

extern "C" std::vector<TypeId> aclnnMultiScaleDeformableAttnGradInferType(std::vector<TypeId> type_ids,
                                                                          AotExtra *extra) {
  std::vector<TypeId> output_type{type_ids[0], type_ids[0], type_ids[0]};
  return output_type;
}

extern "C" std::vector<std::vector<int64_t>> aclnnMultiScaleDeformableAttnGradInferShape(int *ndims, int64_t **shapes,
                                                                                         AotExtra *extra) {
  std::vector<std::vector<int64_t>> res_output_shape;
  auto input0_size = ndims[0];
  std::vector<int64_t> out1_shape;
  for (size_t i = 0; i < input0_size; i++) {
    out1_shape.push_back(shapes[0][i]);
  }
  res_output_shape.emplace_back(out1_shape);

  auto input3_size = ndims[3];
  std::vector<int64_t> out2_shape;
  for (size_t i = 0; i < input3_size; i++) {
    out2_shape.push_back(shapes[3][i]);
  }
  res_output_shape.emplace_back(out2_shape);

  std::vector<int64_t> out3_shape;
  for (size_t i = 0; i < 5; i++) {
    out3_shape.emplace_back(shapes[3][i]);
  }
  res_output_shape.emplace_back(out3_shape);
  return res_output_shape;
}
