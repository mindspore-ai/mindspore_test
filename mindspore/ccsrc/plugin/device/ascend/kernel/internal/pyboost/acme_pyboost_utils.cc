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

#include "plugin/device/ascend/kernel/internal/pyboost/acme_pyboost_utils.h"
namespace mindspore::kernel {
namespace {

void GatherType(const device::DeviceAddressPtr &device_address) {
  if (device_address == nullptr) {
    transform::MemcpyToBuf("None", kSizeFive);
    return;
  }

  // data type
  auto dtype = device_address->type_id();
  transform::MemcpyToBuf(&dtype, sizeof(int));
}

void GatherShape(const device::DeviceAddressPtr &device_address) {
  if (device_address == nullptr) {
    return;
  }

  const auto &shape = device_address->GetShapeVector();
  const auto shape_size = shape.size();
  // view shape
  if (!shape.empty()) {
    transform::MemcpyToBuf(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  }

  const auto &storage_info = device_address->address_common()->tensor_storage_info_;
  if (storage_info != nullptr) {
    // strides
    transform::MemcpyToBuf(storage_info->strides.data(), static_cast<int64_t>(storage_info->strides.size() * sizeof(int64_t)));

    // offset
    transform::MemcpyToBuf(&storage_info->storage_offset, sizeof(int64_t));

    // origin shape
    transform::MemcpyToBuf(storage_info->ori_shape.data(), static_cast<int64_t>(storage_info->ori_shape.size()) * sizeof(int64_t));
  }
}

void GatherType(const mindspore::tensor::BaseTensorPtr &tensor) {
  if (tensor == nullptr) {
    return;
  }

  // "t" for tensor
  transform::MemcpyToBuf("t", 1);

  // data type
  auto dtype = tensor->data_type();
  transform::MemcpyToBuf(&dtype, sizeof(int));

  // storage shape(current hasn't special format)
}

void GatherShape(const mindspore::tensor::BaseTensorPtr &tensor) {
  if (tensor == nullptr) {
    return;
  }

  // "t" for tensor
  transform::MemcpyToBuf("t", 1);

  const auto &shape = tensor->shape();
  const auto shape_size = shape.size();
  // view shape
  if (!shape.empty()) {
    transform::MemcpyToBuf(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  }

  auto storage_info = tensor->storage_info();
  if (storage_info != nullptr) {
    // strides
    transform::MemcpyToBuf(storage_info->strides.data(), static_cast<int64_t>(storage_info->strides.size() * sizeof(int64_t)));

    // offset
    transform::MemcpyToBuf(&storage_info->storage_offset, sizeof(int64_t));

    // origin shape
    transform::MemcpyToBuf(storage_info->ori_shape.data(), static_cast<int64_t>(storage_info->ori_shape.size()) * sizeof(int64_t));
  }
}

}  // namespace

void GatherOpHash(const device::DeviceAddressPtr &device_address) { GatherType(device_address); }

void GatherTilingHash(const device::DeviceAddressPtr &device_address) { GatherShape(device_address); }

void GatherOpHash(const mindspore::tensor::BaseTensorPtr &tensor) { GatherType(tensor); }

void GatherTilingHash(const mindspore::tensor::BaseTensorPtr &tensor) { GatherShape(tensor); }

void GatherOpHash(const std::optional<tensor::BaseTensorPtr> &tensor) {
  // "ot" for optional tensor
  transform::MemcpyToBuf("ot", kSizeTwo);
  if (tensor.has_value()) {
    GatherOpHash(tensor.value());
  }
}

void GatherTilingHash(const std::optional<tensor::BaseTensorPtr> &tensor) {
  if (tensor.has_value()) {
    GatherTilingHash(tensor.value());
  }
}

void GatherOpHash(const std::vector<tensor::BaseTensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    GatherOpHash(tensor);
  }
}

void GatherTilingHash(const std::vector<tensor::BaseTensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    GatherTilingHash(tensor);
  }
}

void GatherHash(const std::vector<int64_t> &int_arrays) { transform::MemcpyToBuf(&int_arrays, sizeof(void *)); }

void GatherOpHash(const std::vector<int64_t> &int_arrays) { GatherHash(int_arrays); }

void GatherTilingHash(const std::vector<int64_t> &int_arrays) { GatherHash(int_arrays); }

void GatherHash(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    transform::MemcpyToBuf("None", kSizeFive);
    return;
  }
  // "s" for scalar
  transform::MemcpyToBuf("s", 1);
  if (scalar->isa<BoolImm>()) {
    auto value = GetValue<bool>(scalar);
    transform::MemcpyToBuf(&value, sizeof(bool));
  } else if (scalar->isa<Int64Imm>()) {
    auto value = GetValue<int64_t>(scalar);
    transform::MemcpyToBuf(&value, sizeof(int64_t));
  } else if (scalar->isa<FP32Imm>()) {
    auto value = GetValue<float>(scalar);
    transform::MemcpyToBuf(&value, sizeof(float));
  } else if (scalar->isa<Int32Imm>()) {
    auto value = GetValue<int32_t>(scalar);
    transform::MemcpyToBuf(&value, sizeof(int32_t));
  } else if (scalar->isa<Int8Imm>()) {
    auto value = GetValue<int8_t>(scalar);
    transform::MemcpyToBuf(&value, sizeof(int8_t));
  } else if (scalar->isa<Int16Imm>()) {
    auto value = GetValue<int16_t>(scalar);
    transform::MemcpyToBuf(&value, sizeof(int16_t));
  } else if (scalar->isa<UInt8Imm>()) {
    auto value = GetValue<uint8_t>(scalar);
    transform::MemcpyToBuf(&value, sizeof(uint8_t));
  } else if (scalar->isa<FP64Imm>()) {
    auto value = GetValue<double>(scalar);
    transform::MemcpyToBuf(&value, sizeof(double));
  } else if (scalar->isa<BF16Imm>()) {
    auto value = GetValue<bfloat16>(scalar);
    transform::MemcpyToBuf(&value, sizeof(int16_t));
  } else {
    MS_LOG(EXCEPTION) << "Currently not support value: " << scalar->ToString();
  }
}

void GatherOpHash(const ScalarPtr &scalar) { GatherHash(scalar); }

void GatherTilingHash(const ScalarPtr &scalar) { GatherHash(scalar); }

void GatherHash(const std::optional<ScalarPtr> &scalar) {
  if (scalar.has_value()) {
    GatherHash(scalar.value());
  } else {
    transform::MemcpyToBuf("None", kSizeFive);
  }
}

void GatherOpHash(const std::optional<ScalarPtr> &scalar) { GatherHash(scalar); }

void GatherTilingHash(const std::optional<ScalarPtr> &scalar) { GatherHash(scalar); }

void GatherHash(const TypePtr &type) {
  const auto type_id = type->type_id();
  transform::MemcpyToBuf(&type_id, sizeof(int));
}

void GatherOpHash(const TypePtr &type) { GatherHash(type); }

void GatherTilingHash(const TypePtr &type) { GatherHash(type); }

void GatherHash(const std::optional<TypePtr> &type) {
  if (type.has_value()) {
    GatherHash(type.value());
  }
}

void GatherOpHash(const std::optional<TypePtr> &type) { GatherHash(type); }

void GatherTilingHash(const std::optional<TypePtr> &type) { GatherHash(type); }

void GatherHash(const string &s) { transform::MemcpyToBuf(s.c_str(), static_cast<int64_t>(s.size())); }

void GatherOpHash(const string &s) { GatherHash(s); }

void GatherTilingHash(const string &s) { GatherHash(s); }

void GatherHash(const std::optional<string> &s) {
  if (s.has_value()) {
    GatherHash(s.value());
  }
}
void GatherOpHash(const std::optional<string> &s) { GatherHash(s); }

void GatherTilingHash(const std::optional<string> &s) { GatherHash(s); }

void GatherOpHash() {}

void GatherTilingHash() {}
}  // namespace mindspore::kernel
