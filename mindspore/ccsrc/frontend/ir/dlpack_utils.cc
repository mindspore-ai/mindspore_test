/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "frontend/ir/dlpack_utils.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/kernel.h"
#include "utils/log_adapter.h"
#include "ir/tensor_storage_info.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace tensor {
namespace {
bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides) {
  if (shape.size() == 0) {
    return true;
  }
  if (shape.size() != strides.size()) {
    MS_LOG(EXCEPTION) << "shape.size() != strides.size()";
  }

  int64_t z = 1;
  for (int64_t i = SizeToLong(shape.size() - 1); i >= 0; --i) {
    const auto &shape_i = shape[i];
    if (shape_i != 1) {
      if (strides[i] == z) {
        z *= shape_i;
      } else {
        return false;
      }
    }
  }
  return true;
}

std::pair<ShapeVector, std::vector<int64_t>> GetOriShapeAndStrides(const ShapeVector &shape,
                                                                   const std::vector<int64_t> &strides) {
  MS_EXCEPTION_IF_CHECK_FAIL(shape.size() == strides.size(), "shape size should be equal to strides size");
  ShapeVector ori_shape;
  int64_t max_loc = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    max_loc += strides[i] * shape[i];
  }
  ori_shape.push_back(max_loc);
  std::vector<int64_t> ori_strides;
  ori_strides.push_back(1);
  return std::make_pair(ori_shape, ori_strides);
}
}  // namespace
DLDataType DLPackUtils::GetDLDataType(const TypeId &type_id) {
  DLDataType dtype;
  switch (type_id) {
    case kNumberTypeFloat32:
      dtype.code = kDLFloat;
      dtype.bits = 32;
      dtype.lanes = 1;
      break;
    case kNumberTypeFloat64:
      dtype.code = kDLFloat;
      dtype.bits = 64;
      dtype.lanes = 1;
      break;
    case kNumberTypeFloat16:
      dtype.code = kDLFloat;
      dtype.bits = 16;
      dtype.lanes = 1;
      break;
    case kNumberTypeBFloat16:
      dtype.code = kDLBfloat;
      dtype.bits = 16;
      dtype.lanes = 1;
      break;
    case kNumberTypeInt8:
      dtype.code = kDLInt;
      dtype.bits = 8;
      dtype.lanes = 1;
      break;
    case kNumberTypeInt16:
      dtype.code = kDLInt;
      dtype.bits = 16;
      dtype.lanes = 1;
      break;
    case kNumberTypeInt32:
      dtype.code = kDLInt;
      dtype.bits = 32;
      dtype.lanes = 1;
      break;
    case kNumberTypeInt64:
      dtype.code = kDLInt;
      dtype.bits = 64;
      dtype.lanes = 1;
      break;
    case kNumberTypeUInt8:
      dtype.code = kDLUInt;
      dtype.bits = 8;
      dtype.lanes = 1;
      break;
    case kNumberTypeUInt16:
      dtype.code = kDLUInt;
      dtype.bits = 16;
      dtype.lanes = 1;
      break;
    case kNumberTypeUInt32:
      dtype.code = kDLUInt;
      dtype.bits = 32;
      dtype.lanes = 1;
      break;
    case kNumberTypeUInt64:
      dtype.code = kDLUInt;
      dtype.bits = 64;
      dtype.lanes = 1;
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported data type: " << type_id;
  }
  return dtype;
}

DLDevice DLPackUtils::GetDLDevice(size_t device_id) {
  // only support Ascend now.
  DLDevice ctx;
  ctx.device_id = static_cast<int32_t>(device_id);
  ctx.device_type = DLDeviceType::kDLExtDev;
  return ctx;
}

TypeId DLPackUtils::GetTypeId(const DLDataType &dtype) {
  TypeId type_id = kTypeUnknown;
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 32) {
      type_id = kNumberTypeFloat32;
    } else if (dtype.bits == 64) {
      type_id = kNumberTypeFloat64;
    } else if (dtype.bits == 16) {
      type_id = kNumberTypeFloat16;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported float bits: " << dtype.bits;
    }
  } else if (dtype.code == kDLBfloat) {
    type_id = kNumberTypeBFloat16;
  } else if (dtype.code == kDLInt) {
    if (dtype.bits == 8) {
      type_id = kNumberTypeInt8;
    } else if (dtype.bits == 16) {
      type_id = kNumberTypeInt16;
    } else if (dtype.bits == 32) {
      type_id = kNumberTypeInt32;
    } else if (dtype.bits == 64) {
      type_id = kNumberTypeInt64;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported int bits: " << dtype.bits;
    }
  } else if (dtype.code == kDLUInt) {
    if (dtype.bits == 8) {
      type_id = kNumberTypeUInt8;
    } else if (dtype.bits == 16) {
      type_id = kNumberTypeUInt16;
    } else if (dtype.bits == 32) {
      type_id = kNumberTypeUInt32;
    } else if (dtype.bits == 64) {
      type_id = kNumberTypeUInt64;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported uint bits: " << dtype.bits;
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type code: " << dtype.code;
  }
  return type_id;
}

TensorPtr DLPackUtils::FromDLPack(DLManagedTensor *dlpack) {
  if (dlpack == nullptr) {
    MS_LOG(EXCEPTION) << "Input dlpack is nullptr";
  }
  // This tensor is treated as a leaf node in the computation graph.
  // As a result, gradient propagation (backpropagation) to other frameworks is not supported.
  auto type_id = GetTypeId(dlpack->dl_tensor.dtype);
  auto shape = dlpack->dl_tensor.shape;
  size_t ndim = static_cast<size_t>(dlpack->dl_tensor.ndim);
  ShapeVector shape_vec;
  for (size_t i = 0; i < ndim; i++) {
    shape_vec.push_back(shape[i]);
  }
  auto strides = dlpack->dl_tensor.strides;
  std::vector<int64_t> strides_vec;
  for (size_t i = 0; i < ndim; i++) {
    strides_vec.push_back(strides[i]);
  }
  auto offset = dlpack->dl_tensor.byte_offset;
  if (offset != 0) {
    MS_LOG(EXCEPTION) << "Unsupported dlpack byte offset: " << offset;
  }
  auto [ori_shape, ori_strides] = GetOriShapeAndStrides(shape_vec, strides_vec);
  auto storage_info = std::make_shared<TensorStorageInfo>(shape_vec, strides_vec, offset, ori_shape, ori_strides,
                                                          IsContiguous(shape_vec, strides_vec));
  auto tensor = std::make_shared<Tensor>(type_id, shape_vec);
  tensor->set_need_pipeline_sync(true);
  tensor->set_storage_info(storage_info);

  auto dldevice = dlpack->dl_tensor.device;
  auto device_id = dldevice.device_id;
  auto device_type = dldevice.device_type;
  if (device_type != kDLExtDev) {
    MS_LOG(EXCEPTION) << "Unsupported device type: " << device_type;
  }

  // only support Ascend now.
  const auto &ms_device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_device != kAscendDevice) {
    MS_LOG(EXCEPTION) << "Only support Ascend device now, but got " << ms_device;
  }
  const auto &ms_device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (ms_device_id != static_cast<uint32_t>(device_id)) {
    MS_LOG(EXCEPTION) << "Device id not match, expect " << ms_device_id << ", but got " << device_id;
  }
  auto device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({kAscendDevice, ms_device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  auto address_size = GetTypeByte(TypeIdToType(type_id)) * SizeOf(ori_shape);
  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, address_size, storage_info->shape, DEFAULT_FORMAT, type_id,
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_, stream_id);
  device_address->set_device_shape(ori_shape);
  device_address->set_tensor_storage_info(storage_info);
  tensor->set_device_address(device_address);
  tensor->set_contiguous_callback([](const DeviceSyncPtr &device_address) -> DeviceSyncPtr {
    MS_EXCEPTION_IF_NULL(device_address);
    auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device_address);
    MS_EXCEPTION_IF_NULL(device_addr);
    // as_numpy sync promise contiguous run_sync
    return runtime::DeviceAddressUtils::ConvertContiguousDeviceAddress(nullptr, device_addr, true);
  });

  // set data
  auto data = dlpack->dl_tensor.data;
  device_address->set_ptr(data);
  device_address->set_from_mem_pool(false);

  // update deleter
  auto ref_cnt = device_address->pointer_ref_count();
  ref_cnt->set_deleter([dlpack = dlpack](void *, bool) {
    if (dlpack == nullptr) {
      return;
    }
    if (dlpack->deleter) {
      dlpack->deleter(dlpack);
    }
  });
  return tensor;
}

namespace {
struct DLMTensor {
  PointerRefCountPtr handle;
  DLManagedTensor tensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};
}  // namespace

static void deleter(DLManagedTensor *arg) { delete static_cast<DLMTensor *>(arg->manager_ctx); }

DLManagedTensor *DLPackUtils::ToDLPack(const Tensor &src) {
  DLMTensor *dlm_tensor = new DLMTensor();
  dlm_tensor->shape = src.shape();
  dlm_tensor->strides = src.stride();
  // normalized strides
  for (size_t i = 0; i < dlm_tensor->shape.size(); i++) {
    if (dlm_tensor->shape[i] < 2) {
      dlm_tensor->strides[i] = 1;
    }
  }

  dlm_tensor->tensor.manager_ctx = dlm_tensor;
  dlm_tensor->tensor.deleter = &deleter;
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(src.device_address());
  if (device_address == nullptr) {
    MS_LOG(EXCEPTION) << "Device address is nullptr";
  }
  dlm_tensor->handle = device_address->pointer_ref_count();
  const auto &ms_device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_device != kAscendDevice) {
    MS_LOG(EXCEPTION) << "Only support Ascend device now, but got " << ms_device;
  }
  const auto &ms_device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  dlm_tensor->tensor.dl_tensor.device = GetDLDevice(ms_device_id);
  dlm_tensor->tensor.dl_tensor.ndim = static_cast<int32_t>(dlm_tensor->shape.size());
  dlm_tensor->tensor.dl_tensor.dtype = GetDLDataType(src.data_type());
  dlm_tensor->tensor.dl_tensor.shape = dlm_tensor->shape.data();
  dlm_tensor->tensor.dl_tensor.strides = dlm_tensor->strides.data();
  dlm_tensor->tensor.dl_tensor.byte_offset = 0;
  runtime::Pipeline::Get().WaitForward();
  dlm_tensor->tensor.dl_tensor.data = const_cast<void *>(device_address->GetPtr());
  if (dlm_tensor->tensor.dl_tensor.data == nullptr) {
    MS_LOG(EXCEPTION) << "Data is nullptr";
  }
  return &(dlm_tensor->tensor);
}
}  // namespace tensor
}  // namespace mindspore
