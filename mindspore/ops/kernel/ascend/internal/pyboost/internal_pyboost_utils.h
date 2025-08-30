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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNALPYBOOST_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNALPYBOOST_UTILS_H_

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "kernel/ascend/internal/internal_helper.h"

namespace mindspore::kernel {
namespace transform = device::ascend;
BACKEND_EXPORT void GatherOpHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherOpHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const std::vector<int64_t> &);

template <typename T>
BACKEND_EXPORT void GatherOpHash(const T &value) {
  transform::MemcpyToBuf(&value, sizeof(T));
}

template <typename T>
BACKEND_EXPORT void GatherOpHash(std::optional<T> value) {
  if (value.has_value()) {
    GatherOpHash(value.value());
  }
}

BACKEND_EXPORT void GatherOpHash(const std::string &);
BACKEND_EXPORT void GatherOpHash(const std::optional<string> &);

BACKEND_EXPORT void GatherOpHash(const ScalarPtr &);
BACKEND_EXPORT void GatherOpHash(const std::optional<ScalarPtr> &);

BACKEND_EXPORT void GatherOpHash(const TypePtr &);
BACKEND_EXPORT void GatherOpHash(const std::optional<TypePtr> &);

template <typename T>
BACKEND_EXPORT void GatherOpHash(const std::vector<T> &values) {
  transform::MemcpyToBuf(reinterpret_cast<const void *>(values.data()), values.size() * sizeof(T));
}

BACKEND_EXPORT void GatherOpHash();

template <typename T, typename... Args>
void GatherOpHash(const T &arg, const Args &... args) {
  GatherOpHash(arg);
  GatherOpHash(args...);
}

template <typename... Args>
uint64_t CalcInternalOpApiHash(const std::string &arg, const Args &... args) {
  transform::g_hash_offset = 0;
  GatherOpHash(arg, args...);
  return transform::calc_hash_id();
}

BACKEND_EXPORT void GatherTilingHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherTilingHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const std::vector<int64_t> &);

template <typename T>
BACKEND_EXPORT void GatherTilingHash(const T &value) {
  GatherOpHash(value);
}

BACKEND_EXPORT void GatherTilingHash();

template <typename T, typename... Args>
void GatherTilingHash(const T &arg, const Args &... args) {
  GatherTilingHash(arg);
  GatherTilingHash(args...);
}

template <typename... Args>
uint64_t CalcInternalOpTilingHash(const std::string &arg, const Args &... args) {
  GatherTilingHash(arg, args...);
  return transform::calc_hash_id();
}

template <typename D, typename S>
void ConvertVectorDtype(std::vector<D> *dst_vec, const std::vector<S> &src_vec) {
  dst_vec->clear();
  for (const auto &item : src_vec) {
    dst_vec->emplace_back(static_cast<D>(item));
  }
}

template <typename T>
ValuePtr ConvertValue(const std::optional<T> &t) {
  if (t.has_value()) {
    return t.value();
  }
  return mindspore::kNone;
}

template <typename T>
ValuePtr ConvertValue(const T &t) {
  return t;
}

}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNALPYBOOST_UTILS_H_
