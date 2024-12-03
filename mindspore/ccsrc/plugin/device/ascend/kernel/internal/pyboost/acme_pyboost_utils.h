/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_PYBOOST_CACHE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_PYBOOST_CACHE_H_

#include <string>
#include <vector>
#include <utility>
#include "transform/acl_ir/op_api_cache.h"

namespace mindspore::kernel {
BACKEND_EXPORT void GatherOpHash(const device::DeviceAddressPtr &);
BACKEND_EXPORT void GatherOpHash(const mindspore::tensor::BaseTensorPtr &);
BACKEND_EXPORT void GatherOpHash(const std::optional<tensor::BaseTensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const std::vector<tensor::BaseTensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherOpHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherOpHash(const std::vector<int64_t> &);

BACKEND_EXPORT void GatherTilingHash(const device::DeviceAddressPtr &);
BACKEND_EXPORT void GatherTilingHash(const mindspore::tensor::BaseTensorPtr &);
BACKEND_EXPORT void GatherTilingHash(const std::optional<tensor::BaseTensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const std::vector<tensor::BaseTensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherTilingHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherTilingHash(const std::vector<int64_t> &);


template <typename T>
BACKEND_EXPORT void GatherOpHash(const T &value) {
  GatherOpHash(value);
}

template <typename T>
BACKEND_EXPORT void GatherTilingHash(const T &value) {
  GatherTilingHash(value);
}

BACKEND_EXPORT void GatherOpHash();
BACKEND_EXPORT void GatherTilingHash();

template <typename T, typename... Args>
void GatherOpHash(const T &arg, const Args &... args) {
  GatherOpHash(arg);
  GatherOpHash(args...);
}

template <typename T, typename... Args>
void GatherTilingHash(const T &arg, const Args &... args) {
  GatherTilingHash(arg);
  GatherTilingHash(args...);
}

// 创建acme算子主要看输入的数据类型和属性
template <typename... Args>
uint64_t CalcAcmeOpApiHash(const std::string &arg, const Args &... args) {
  transform::g_hash_offset = 0;
  GatherOpHash(arg, args...);
  return transform::calc_hash_id();
}

// acme算子tiling还需要包含输入的shape和属性是否变化
template <typename... Args>
uint64_t CalcAcmeOpTilingHash(const std::string &arg, const Args &... args) {
  GatherTilingHash(arg, args...);
  return transform::calc_hash_id();
}

}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_PYBOOST_CACHE_H_
