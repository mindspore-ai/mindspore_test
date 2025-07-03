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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_HOST_VALUE_STORE_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_HOST_VALUE_STORE_H_

#include <memory>
#include <unordered_map>

#include "ir/anf.h"
#include "mindspore/core/include/base/base.h"

#include "dalang/dair/tensor/tensor.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {
class HostValueStore {
 public:
  static HostValueStore &GetInstance() {
    static HostValueStore instance;
    return instance;
  }

  HostValueStore(const HostValueStore &) = delete;
  HostValueStore &operator=(const HostValueStore &) = delete;
  ~HostValueStore() = default;

  void Insert(da::tensor::DATensor *k, const ValuePtr &v) {
    if (host_da_tensor_value_.find(k) != host_da_tensor_value_.end()) {
      MS_LOG(EXCEPTION) << "Duplicate insert for DATensor: " << k;
    }

    if (k->tensorType != da::tensor::TensorType::HOST_TENSOR) {
      MS_LOG(EXCEPTION) << "DATensor is not host value: " << k;
    }

    MS_LOG(INFO) << "Insert host value for DATensor: " << k << ", value: " << v->ToString();
    host_da_tensor_value_[k] = v;
  }

  ValuePtr &Get(da::tensor::DATensor *k) {
    auto iter = host_da_tensor_value_.find(k);
    if (iter == host_da_tensor_value_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find host value store for DATensor: " << k;
    }
    return iter->second;
  }

  std::unordered_map<da::tensor::DATensor *, ValuePtr> &GetHostValueMap() { return host_da_tensor_value_; }
  void Clear() { host_da_tensor_value_.clear(); }

 private:
  HostValueStore() = default;

  std::unordered_map<da::tensor::DATensor *, ValuePtr> host_da_tensor_value_;
};
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_HOST_VALUE_STORE_H_
