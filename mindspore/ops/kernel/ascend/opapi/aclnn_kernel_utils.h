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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_KERNEL_UTILS_H
#define MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_KERNEL_UTILS_H
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <utility>
#include "ops/base_operator.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "mindspore/core/include/ir/tensor.h"

namespace mindspore {
namespace kernel {

template <size_t N, std::size_t... Is>
auto GetTupleFrontImpl(const std::vector<tensor::TensorPtr> &vecs, std::index_sequence<Is...>) {
  return std::make_tuple(vecs[Is]...);
}

template <size_t N>
auto GetTupleFront(const std::vector<tensor::TensorPtr> &vecs) {
  return GetTupleFrontImpl<N>(vecs, std::make_index_sequence<N>());
}

template <size_t N, std::size_t... Is>
auto GetTupleFrontImpl(const std::vector<KernelTensor *> &vecs, std::index_sequence<Is...>) {
  return std::make_tuple(vecs[Is]...);
}

template <size_t N>
auto GetTupleFront(const std::vector<KernelTensor *> &vecs) {
  return GetTupleFrontImpl<N>(vecs, std::make_index_sequence<N>());
}

template <size_t N, std::size_t... Is>
auto GetTupleFrontImpl(const std::vector<ValuePtr> &vecs, std::index_sequence<Is...>) {
  return std::make_tuple(vecs[Is]...);
}

template <size_t N>
auto GetTupleFront(const std::vector<ValuePtr> &vecs) {
  return GetTupleFrontImpl<N>(vecs, std::make_index_sequence<N>());
}

template <typename T, typename... Vecs>
std::vector<T> ConcatVecs(const std::vector<T> &vec, const Vecs &... vecs) {
  std::vector<T> result = vec;
  (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
  return result;
}

template <typename T, typename... Vecs>
std::vector<T> ConcatVecs(const Vecs &... vecs) {
  static_assert((std::is_same_v<T, typename Vecs::value_type> && ...), "All vectors must have the same type!");
  std::vector<T> result;
  (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
  return result;
}

template <size_t N, typename... Ts>
auto GetKernelTuple(const std::vector<Ts> &... vecs) {
  const auto &new_vec = ConcatVecs(vecs...);
  if (new_vec.size() != N) {
    MS_LOG(EXCEPTION) << "Config op input and output's size must be same, but get " << N << " with " << new_vec.size();
  }
  const auto &result = GetTupleFront<N>(new_vec);
  return result;
}

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_KERNEL_UTILS_H
