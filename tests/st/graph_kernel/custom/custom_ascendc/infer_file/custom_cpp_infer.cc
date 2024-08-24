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