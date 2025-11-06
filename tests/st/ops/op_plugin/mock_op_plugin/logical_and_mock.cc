/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

extern "C" {
// Mock implementation of the logical_and operator.
// Not fully implemented, only for certain test cases.
int LogicalAnd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
               void *extra) {
  std::cout << "op_plugin mock: LogicalAnd called" << std::endl;
  constexpr int expected_nparam = 3;
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for LogicalAnd operator" << std::endl;
    return -1;
  }

  const bool *x = static_cast<const bool *>(params[0]);
  const bool *y = static_cast<const bool *>(params[1]);
  bool *out = static_cast<bool *>(params[2]);

  int dims = ndims[0];
  if (dims < 0) {
    std::cout << "Invalid dims for LogicalAnd operator: " << dims << std::endl;
    return -1;
  }
  if (ndims[1] != dims || ndims[2] != dims) {
    std::cout << "Invalid ndims for LogicalAnd operator" << std::endl;
    return -1;
  }

  size_t numel = 1;
  for (int i = 0; i < dims; ++i) {
    int64_t d0 = shapes[0][i];
    int64_t d1 = shapes[1][i];
    int64_t d2 = shapes[2][i];
    if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
      std::cout << "Invalid shapes for LogicalAnd operator: d0 <= 0 || d1 <= 0 || d2 <= 0" << std::endl;
      return -1;
    }
    if (d0 != d1 || d0 != d2) {
      std::cout << "Invalid shapes for LogicalAnd operator: d0 != d1 || d0 != d2" << std::endl;
      return -1;
    }
    numel *= static_cast<size_t>(d0);
  }

  for (size_t i = 0; i < numel; ++i) {
    out[i] = (x[i] || y[i]);  // wrong implementation by purpose to ensure op plugin is used
  }
  return 0;
}

}  // extern "C"
