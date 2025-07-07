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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_DEVICE_TENSOR_STORE_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_DEVICE_TENSOR_STORE_H_

#include <memory>
#include <unordered_map>

#include "ir/anf.h"
#include "mindspore/core/include/base/base.h"

#include "dalang/dair/tensor/tensor.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {
class DeviceTensorStore {
 public:
  static DeviceTensorStore &GetInstance() {
    static DeviceTensorStore instance;
    return instance;
  }

  DeviceTensorStore(const DeviceTensorStore &) = delete;
  DeviceTensorStore &operator=(const DeviceTensorStore &) = delete;
  ~DeviceTensorStore() = default;

  void Insert(const ValuePtr &k, da::tensor::DATensor *v) {
    if (device_da_tensor_.find(k) != device_da_tensor_.end()) {
      MS_LOG(EXCEPTION) << "Duplicate insert for " << k->ToString();
    }

    if (v->tensorType != da::tensor::TensorType::DEVICE_TENSOR) {
      MS_LOG(EXCEPTION) << "Expect a device DATensor, but got tensorType: " << v->tensorType;
    }

    MS_LOG(INFO) << "Insert device tensor for DATensor: " << v << ", value: " << k->ToString();
    device_da_tensor_[k] = v;
  }

  da::tensor::DATensor *Get(const ValuePtr &k) {
    auto iter = device_da_tensor_.find(k);
    if (iter == device_da_tensor_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find DATensor device store for value: " << k->ToString();
    }
    return iter->second;
  }

  bool HasValue(const ValuePtr &value) const { return device_da_tensor_.find(value) != device_da_tensor_.end(); }

  void Clear() { device_da_tensor_.clear(); }

 private:
  DeviceTensorStore() = default;

  std::unordered_map<ValuePtr, da::tensor::DATensor *> device_da_tensor_;
};
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_DEVICE_TENSOR_STORE_H_
