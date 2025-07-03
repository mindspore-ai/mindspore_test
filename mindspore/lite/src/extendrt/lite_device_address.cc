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

#include "src/extendrt/lite_device_address.h"

#include <string>
#include <utility>

#include "ir/device_address_maker.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace runtime {
namespace test {
namespace {
DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                                     TypeId type_id, const std::string &device_name, uint32_t device_id,
                                     uint32_t stream_id, const UserDataPtr &user_data = nullptr) {
  return std::make_shared<TestDeviceAddress>(ptr, size, "falut", type_id, device_name, 0);
}
DeviceSyncPtr MakeTestDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                    DeviceAddressDeleter &&deleter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_address =
    CreateDeviceAddress(data_ptr, data_size, shape, Format::DEFAULT_FORMAT, data_type, "CPU", device_id, 0);
  device_address->SetPointerRefCountDeleter(std::move(deleter));
  return device_address;
}

const char device_name[] = "CPU";
REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kCPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeTestDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});

template <typename T>
void CopyData(T *src_ptr, size_t size, void *dst_ptr, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(dst_ptr);
  switch (type_id) {
    case kNumberTypeBool: {
      auto buf = static_cast<bool *>(dst_ptr);
      return tensor::TransDataType<bool>(src_ptr, buf, size);
    }
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(dst_ptr);
      return tensor::TransDataType<uint8_t>(src_ptr, buf, size);
    }
    case kNumberTypeInt4: {
      auto buf = static_cast<int8_t *>(dst_ptr);
      return tensor::TransDataType<int8_t>(src_ptr, buf, size);
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(dst_ptr);
      return tensor::TransDataType<int8_t>(src_ptr, buf, size);
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(dst_ptr);
      return tensor::TransDataType<int16_t>(src_ptr, buf, size);
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(dst_ptr);
      return tensor::TransDataType<int32_t>(src_ptr, buf, size);
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(dst_ptr);
      return tensor::TransDataType<int64_t>(src_ptr, buf, size);
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(dst_ptr);
      return tensor::TransDataType<uint16_t>(src_ptr, buf, size);
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(dst_ptr);
      return tensor::TransDataType<uint32_t>(src_ptr, buf, size);
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(dst_ptr);
      return tensor::TransDataType<uint64_t>(src_ptr, buf, size);
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(dst_ptr);
      return tensor::TransDataType<float16>(src_ptr, buf, size);
    }
    case kNumberTypeFloat8E4M3FN: {
      auto buf = static_cast<float8_e4m3fn *>(dst_ptr);
      return tensor::TransDataType<float8_e4m3fn>(src_ptr, buf, size);
    }
    case kNumberTypeFloat8E5M2: {
      auto buf = static_cast<float8_e5m2 *>(dst_ptr);
      return tensor::TransDataType<float8_e5m2>(src_ptr, buf, size);
    }
    case kNumberTypeHiFloat8: {
      auto buf = static_cast<hifloat8 *>(dst_ptr);
      return tensor::TransDataType<hifloat8>(src_ptr, buf, size);
    }
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16: {
      auto buf = static_cast<bfloat16 *>(dst_ptr);
      return tensor::TransDataType<bfloat16>(src_ptr, buf, size);
    }
#endif
    case kNumberTypeComplex64: {
      auto buf = static_cast<ComplexStorage<float> *>(dst_ptr);
      return tensor::TransDataType<ComplexStorage<float>>(src_ptr, buf, size);
    }
    case kNumberTypeComplex128: {
      auto buf = static_cast<ComplexStorage<double> *>(dst_ptr);
      return tensor::TransDataType<ComplexStorage<double>>(src_ptr, buf, size);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported dst data type: " << type_id << ".";
}

void CopyData(const DeviceAddress *src_device_address, const DeviceAddress *dst_device_address) {
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_EXCEPTION_IF_NULL(dst_device_address);
  if (src_device_address->GetShapeVector() != dst_device_address->GetShapeVector()) {
    MS_LOG(EXCEPTION) << "Not same shape in device address:" << src_device_address->ToString()
                      << " and:" << dst_device_address->ToString();
  }
  const size_t size = SizeOf(src_device_address->GetShapeVector());
  auto src_ptr = src_device_address->GetMutablePtr();
  auto dst_ptr = dst_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(dst_ptr);
  switch (src_device_address->type_id()) {
    case kNumberTypeBool: {
      auto buf = static_cast<bool *>(src_ptr);
      return CopyData<bool>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(src_ptr);
      return CopyData<uint8_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeInt4: {
      auto buf = static_cast<int8_t *>(src_ptr);
      return CopyData<int8_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(src_ptr);
      return CopyData<int8_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(src_ptr);
      return CopyData<int16_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(src_ptr);
      return CopyData<int32_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(src_ptr);
      return CopyData<int64_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(src_ptr);
      return CopyData<uint16_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(src_ptr);
      return CopyData<uint32_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(src_ptr);
      return CopyData<uint64_t>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(src_ptr);
      return CopyData<float16>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeFloat8E4M3FN: {
      auto buf = static_cast<float8_e4m3fn *>(src_ptr);
      return CopyData<float8_e4m3fn>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeFloat8E5M2: {
      auto buf = static_cast<float8_e5m2 *>(src_ptr);
      return CopyData<float8_e5m2>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeHiFloat8: {
      auto buf = static_cast<hifloat8 *>(src_ptr);
      return CopyData<hifloat8>(buf, size, dst_ptr, dst_device_address->type_id());
    }
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16: {
      auto buf = static_cast<bfloat16 *>(src_ptr);
      return CopyData<bfloat16>(buf, size, dst_ptr, dst_device_address->type_id());
    }
#endif
    case kNumberTypeComplex64: {
      auto buf = static_cast<ComplexStorage<float> *>(src_ptr);
      return CopyData<ComplexStorage<float>>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    case kNumberTypeComplex128: {
      auto buf = static_cast<ComplexStorage<double> *>(src_ptr);
      return CopyData<ComplexStorage<double>>(buf, size, dst_ptr, dst_device_address->type_id());
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported src data type: " << src_device_address->type_id()
                    << ".";
}
}  // namespace

bool LiteSyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id) {
  return AsyncCopy(dst_device_sync, src_device_sync, stream_id, false);
}

bool LiteAsyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id, bool) {
  const auto &dst_device_address = dynamic_cast<const TestDeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const TestDeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(INFO) << "No need sync for dst device address: " << dst_device_address->ToString()
                 << " and src device address: " << src_device_address->ToString();
    return true;
  }

  if (dst_device_address->format() != src_device_address->format()) {
    MS_LOG(ERROR) << "Format is different, src(format:" << src_device_address->format()
                  << "), dst(format:" << dst_device_address->format() << ") for device address:" << dst_device_address;
    return false;
  }
  auto dst_ptr = dst_device_address->GetMutablePtr();
  auto src_ptr = src_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_device_address->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetMutablePtr());
  if (dst_ptr == src_ptr) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }
  auto dst_type_id = dst_device_address->type_id();
  auto src_type_id = src_device_address->type_id();

  if (src_type_id == dst_type_id) {
    if (src_device_address->GetSize() > dst_device_address->GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, src size: " << src_device_address->GetSize()
                      << ", dst size: " << dst_device_address->GetSize();
      return true;
    }
    auto ret_code = memcpy_s(dst_ptr, src_device_address->GetSize(), src_ptr, src_device_address->GetSize());
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      device::ConvertSameType(dst_device_address->GetMutablePtr(), src_device_address->GetMutablePtr(),
                              dst_device_address->GetSize(), src_type_id);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor from device address:" << src_device_address->ToString()
                    << " to :" << dst_device_address->ToString();
      return false;
    } else {
      return true;
    }
  }

  if (dst_type_id == kNumberTypeFloat16 && src_type_id == kNumberTypeFloat32) {
    device::FloatToHalf(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeFloat64 && src_type_id == kNumberTypeFloat32) {
    device::FloatToDouble(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(double));
  } else if (dst_type_id == kNumberTypeFloat32 && src_type_id == kNumberTypeFloat64) {
    device::DoubleToFloat(dst_ptr, src_ptr, dst_device_address->GetSize() >> 2);
  } else if (dst_type_id == kNumberTypeInt16 && src_type_id == kNumberTypeInt32) {
    device::IntToShort(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeInt64 && src_type_id == kNumberTypeInt32) {
    device::IntToLong(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(int64_t));
  } else {
    MS_LOG(INFO) << "Types not match. src type: " << TypeIdLabel(src_type_id)
                 << ", dst type: " << TypeIdLabel(dst_type_id) << " device_address:" << dst_device_address << " !";
    CopyData(src_device_address, dst_device_address);
    return true;
  }
  return true;
}

MS_REGISTER_HAL_COPY_FUNC(DeviceType::kCPU,
                          ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                              size_t stream_id) { return LiteSyncCopy(dst_device_sync, src_device_sync, stream_id); }),
                          ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                              size_t stream_id,
                              bool) { return LiteSyncCopy(dst_device_sync, src_device_sync, stream_id); }));

}  // namespace test
}  // namespace runtime
}  // namespace mindspore
