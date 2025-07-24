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

#include <string>
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

  void InsertValueForDATensor(da::tensor::DATensor *k, const ValuePtr &v) {
    MS_EXCEPTION_IF_NULL(k);
    MS_EXCEPTION_IF_NULL(v);
    if (host_da_tensor_value_.find(k) != host_da_tensor_value_.end()) {
      MS_LOG(EXCEPTION) << "Duplicate insert for DATensor: " << k;
    }

    if (k->tensorType != da::tensor::TensorType::HOST_TENSOR) {
      MS_LOG(EXCEPTION) << "DATensor is not host value: " << k;
    }

    MS_LOG(INFO) << "Insert host value for DATensor: " << k << ", value: " << v->ToString();
    host_da_tensor_value_[k] = v;
    da_tensor_host_value_[v] = k;
  }

  ValuePtr &GetValueByDATensor(da::tensor::DATensor *k) {
    MS_EXCEPTION_IF_NULL(k);
    auto iter = host_da_tensor_value_.find(k);
    if (iter == host_da_tensor_value_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find host value store for DATensor: " << k;
    }
    return iter->second;
  }

  da::tensor::DATensor *GetDATensorByValue(const ValuePtr &v) {
    MS_EXCEPTION_IF_NULL(v);
    auto iter = da_tensor_host_value_.find(v);
    if (iter == da_tensor_host_value_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find DATensor for host value: " << v->ToString();
    }
    return iter->second;
  }

  void InsertPrimForDATensor(da::tensor::DATensor *da_tensor, const PrimitivePtr &ms_prim) {
    MS_EXCEPTION_IF_NULL(da_tensor);
    MS_EXCEPTION_IF_NULL(ms_prim);
    if (da_tensor_primitive_.find(da_tensor) != da_tensor_primitive_.end()) {
      MS_LOG(EXCEPTION) << "Duplicate insert primitive for DATensor: " << da_tensor;
    }
    MS_LOG(INFO) << "Insert primitive: " << ms_prim->ToString() << " for DATensor: " << da_tensor;
    da_tensor_primitive_[da_tensor] = ms_prim;
  }

  PrimitivePtr &GetPrimByDATensor(da::tensor::DATensor *da_tensor) {
    MS_EXCEPTION_IF_NULL(da_tensor);
    auto iter = da_tensor_primitive_.find(da_tensor);
    if (iter == da_tensor_primitive_.end()) {
      MS_LOG(EXCEPTION) << "Can not find ms primitive for DATensor: " << da_tensor;
    }
    return iter->second;
  }

  bool HasValue(const ValuePtr &v) const { return da_tensor_host_value_.find(v) != da_tensor_host_value_.end(); }

  std::unordered_map<da::tensor::DATensor *, ValuePtr> &GetHostValueMap() { return host_da_tensor_value_; }
  void Clear() {
    host_da_tensor_value_.clear();
    da_tensor_host_value_.clear();
    da_tensor_primitive_.clear();
  }

 private:
  HostValueStore() = default;

  std::unordered_map<da::tensor::DATensor *, ValuePtr> host_da_tensor_value_;
  std::unordered_map<ValuePtr, da::tensor::DATensor *> da_tensor_host_value_;
  std::unordered_map<da::tensor::DATensor *, PrimitivePtr> da_tensor_primitive_;
};
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_HOST_VALUE_STORE_H_
