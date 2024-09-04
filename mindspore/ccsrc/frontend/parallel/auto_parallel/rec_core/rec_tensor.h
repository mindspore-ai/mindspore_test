/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_
#define PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_

#include <cstdint>
#include <vector>
#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
enum TensorType { kInt8, kFloat16, kFloat32, kDouble64 };

struct Shape4D {
  int64_t shape_n = 1;
  int64_t shape_c = 1;
  int64_t shape_h = 1;
  int64_t shape_w = 1;

  std::vector<int64_t> ShapeToVector() const {
    std::vector<int64_t> shape_vector;
    shape_vector.push_back(shape_n);
    shape_vector.push_back(shape_c);
    shape_vector.push_back(shape_h);
    shape_vector.push_back(shape_w);
    return shape_vector;
  }

  bool operator==(const Shape4D sh) const {
    if (shape_n == sh.shape_n && shape_c == sh.shape_c && shape_h == sh.shape_h && shape_w == sh.shape_w) {
      return true;
    }
    return false;
  }

  friend std::ostream &operator<<(std::ostream &os, Shape4D const &shape) {
    return os << "n: " << shape.shape_n << " c: " << shape.shape_c << " h: " << shape.shape_h
              << " w: " << shape.shape_w;
  }
};

inline Shape4D VectorToShape(std::vector<int64_t> vec) {
  Shape4D vector_shape;
  if (vec.size() != SIZE_FOUR) {
    MS_LOG(WARNING) << "Only a vector with length 4 can be converted to Shape4D";
  } else {
    vector_shape.shape_n = vec[INDEX_ZERO];
    vector_shape.shape_c = vec[INDEX_ONE];
    vector_shape.shape_h = vec[INDEX_TWO];
    vector_shape.shape_w = vec[INDEX_THREE];
  }
  return vector_shape;
}

struct TensorParam {
  TensorType tensor_type = kFloat32;  // default as float.
  Shape4D tensor_shape;
  TensorStr4D tensor_str;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_
