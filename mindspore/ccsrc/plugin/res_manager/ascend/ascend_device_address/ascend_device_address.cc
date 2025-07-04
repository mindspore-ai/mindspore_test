/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_address.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <set>
#include "graph/types.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "plugin/res_manager/ascend/event/ascend_event.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/ms_device_shape_transfer.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace py = pybind11;
namespace mindspore {
namespace device {
namespace ascend {
namespace {
// Create a mutex for stream.
std::mutex *CreateStreamMutex(const void *stream, std::shared_mutex *shd_mtx,
                              mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> *mtxs_for_streams) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  MS_EXCEPTION_IF_NULL(mtxs_for_streams);

  std::unique_lock<std::shared_mutex> unq_lock(*shd_mtx);
  auto ret_pair = mtxs_for_streams->emplace(stream, std::make_shared<std::mutex>());

  MS_EXCEPTION_IF_NULL(ret_pair.first->second);
  return ret_pair.first->second.get();
}

// Check whether mutex exists for a stream.
std::pair<bool, std::mutex *> CheckStreamMutexExist(
  const void *stream, const mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> &mtxs_for_streams,
  std::shared_mutex *shd_mtx) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  std::shared_lock<std::shared_mutex> shd_lock(*shd_mtx);
  auto iter = mtxs_for_streams.find(stream);
  if (iter != mtxs_for_streams.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return std::make_pair(true, iter->second.get());
  }
  return std::make_pair(false, nullptr);
}

std::lock_guard<std::mutex> LockRuntime(const void *stream) {
  MS_EXCEPTION_IF_NULL(stream);
  // Read-write lock for accessing mtxs_for_streams map.
  // When the lock of each stream is created, mtxs_for_streams can be accessed concurrently to improve performance.
  static std::shared_mutex shd_mtx;
  static mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> mtxs_for_streams;

  std::mutex *stream_mtx = nullptr;
  // Check whether mutex exists for a stream.
  std::pair<bool, std::mutex *> ret_pair = CheckStreamMutexExist(stream, mtxs_for_streams, &shd_mtx);
  if (ret_pair.first) {
    stream_mtx = ret_pair.second;
  } else {
    // Create a mutex for stream.
    stream_mtx = CreateStreamMutex(stream, &shd_mtx, &mtxs_for_streams);
  }

  MS_EXCEPTION_IF_NULL(stream_mtx);
  return std::lock_guard<std::mutex>(*stream_mtx);
}

void SetContextForce() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContextForce(device_id);
}

bool SyncStreamUtils() {
  SetContextForce();
  std::set<aclrtStream> except_streams;
  if (AscendStreamMng::GetInstance().default_stream() != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = LockRuntime(AscendStreamMng::GetInstance().default_stream());
    if (!AscendStreamMng::GetInstance().SyncStream(AscendStreamMng::GetInstance().default_stream())) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
    (void)except_streams.insert(AscendStreamMng::GetInstance().default_stream());
  }
  if (AscendStreamMng::GetInstance().communication_stream() != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = LockRuntime(AscendStreamMng::GetInstance().communication_stream());
    if (!AscendStreamMng::GetInstance().SyncStream(AscendStreamMng::GetInstance().communication_stream())) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
    (void)except_streams.insert(AscendStreamMng::GetInstance().communication_stream());
  }

  // Sync all stream except stream_ and communication_stream_.
  if (!AscendStreamMng::GetInstance().SyncExceptStreamsInList(except_streams)) {
    MS_LOG(ERROR) << "Sync except streams failed.";
    return false;
  }
  return true;
}

bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind, void *stream) {
  SetContextForce();
  if (size == 0) {
    MS_LOG(DEBUG) << "rtMemcpyAsync size is 0, copy kind:" << kind;
    return true;
  }
  if (stream == nullptr) {
    MS_LOG(ERROR) << "MemcpyAsync failed. stream is nullptr";
    return false;
  }

  if (dst == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync dst ptr is null, copy kind:" << kind;
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync src ptr is null, copy kind:" << kind;
    return false;
  }
  // cppcheck-suppress unreadVariable
  auto lock = LockRuntime(stream);
  if (!common::IsCompileSimulation()) {
    if (ACL_SUCCESS !=
        CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, static_cast<aclrtMemcpyKind>(kind), stream)) {
      MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error.";
      return false;
    }
  }
  return true;
}
}  // namespace
const auto kFloat16Bytes = 2;
const auto kFloatBytes = sizeof(float);
const auto kFloat64Bytes = 8;
static std::recursive_mutex transdata_mutx;

#if defined(RT_MEMORY_P2PDMA)
static std::mutex dma_lock;
#endif

bool IsUseTransDataTypeFormat(const std::pair<std::string, std::string> &type_format) {
  static const std::set<std::pair<std::string, std::string>> use_trans_data = {
    std::make_pair("float16", mindspore::kOpFormat_NC1HWC0), std::make_pair("float32", mindspore::kOpFormat_NC1HWC0),
    std::make_pair("bool", mindspore::kOpFormat_NC1HWC0),    std::make_pair("float32", mindspore::kOpFormat_FRAC_Z),
    std::make_pair("float16", mindspore::kOpFormat_FRAC_Z),  std::make_pair("float16", mindspore::kOpFormat_FRAC_NZ),
    std::make_pair("float32", mindspore::kOpFormat_FRAC_NZ), std::make_pair("int32", mindspore::kOpFormat_FRAC_NZ),
    std::make_pair("float16", mindspore::kOpFormat_NHWC),    std::make_pair("float32", mindspore::kOpFormat_NHWC),
    std::make_pair("int8", mindspore::kOpFormat_NHWC),       std::make_pair("int16", mindspore::kOpFormat_NHWC),
    std::make_pair("int32", mindspore::kOpFormat_NHWC),      std::make_pair("int64", mindspore::kOpFormat_NHWC),
    std::make_pair("uint8", mindspore::kOpFormat_NHWC),      std::make_pair("uint16", mindspore::kOpFormat_NHWC),
    std::make_pair("uint32", mindspore::kOpFormat_NHWC),     std::make_pair("uint64", mindspore::kOpFormat_NHWC),
    std::make_pair("float16", mindspore::kOpFormat_HWCN),    std::make_pair("float32", mindspore::kOpFormat_HWCN),
    std::make_pair("int8", mindspore::kOpFormat_HWCN),       std::make_pair("int16", mindspore::kOpFormat_HWCN),
    std::make_pair("int32", mindspore::kOpFormat_HWCN),      std::make_pair("int64", mindspore::kOpFormat_HWCN),
    std::make_pair("uint8", mindspore::kOpFormat_HWCN),      std::make_pair("uint16", mindspore::kOpFormat_HWCN),
    std::make_pair("uint32", mindspore::kOpFormat_HWCN),     std::make_pair("uint64", mindspore::kOpFormat_HWCN)};
  return use_trans_data.find(type_format) != use_trans_data.end();
}

static const std::set<std::string> basic_format = {kOpFormat_NCHW, kOpFormat_DEFAULT, kOpFormat_NCDHW, kOpFormat_ND};

bool IsOpNeedTransFormat(const std::string &format) {
  static const std::set<std::string> op_need_trans_format = {
    kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
    kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};
  return op_need_trans_format.find(format) != op_need_trans_format.end();
}

void AscendDeviceAddress::SyncHostMemoryToDeviceWithCopySrc(void *dst, const void *src, uint64_t size,
                                                            aclrtMemcpyKind kind, size_t stream_id) const {
  MS_LOG(DEBUG) << "Begin, size:" << size;
  std::shared_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[size]);
  MS_EXCEPTION_IF_NULL(buffer);
  auto ret_code = memcpy_s(buffer.get(), size, src, size);
  // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
  if (ret_code == ERANGE) {
    ConvertSameType(buffer.get(), src, size, type_id());
  }

  size_t real_stream_id = (stream_id == SIZE_MAX) ? this->stream_id() : stream_id;
  const auto stream = AscendStreamMng::GetInstance().GetStream(real_stream_id);
  auto ret = MemcpyAsync(dst, buffer.get(), size, static_cast<int32_t>(kind), stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "MemcpyAsync failed!";
  }

  std::function<void(void)> callback_func = [buffer]() {
    // Clear buffer automatically.
    MS_LOG(DEBUG) << "callback_func exec, buffer cnt:" << buffer.use_count();
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ResKey res_key{GetDeviceTypeByName(device_name), device_id};
  auto res_manager = HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto callback_ret = res_manager->LaunchCallback(callback_func, real_stream_id);
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "LaunchCallback failed";
  }
}

void AscendDeviceAddress::SyncHostMemoryToDeviceForTensorFromNumpy(void *dst, const void *src, uint64_t size,
                                                                   aclrtMemcpyKind kind) const {
  MS_LOG(DEBUG) << "Begin, size:" << size;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  // Memcpy needs to be synchronized firstm, if tensor data is from numpy.
  const auto stream = AscendStreamMng::GetInstance().GetStream(this->stream_id());
  // cppcheck-suppress unreadVariable
  auto lock = LockRuntime(stream);
  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_EXCEPTION(DeviceProcessError) << "Sync stream error!";
  }

  auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, dst, size, src, size, kind);
  MS_LOG(DEBUG) << "tensor is_from_numpy, sync it first";
  if (ret_rt_memcpy != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
  }
}

void AscendDeviceAddress::SyncHostMemoryToDeviceWithTensorData(void *dst, const void *src, uint64_t size,
                                                               aclrtMemcpyKind kind,
                                                               const tensor::TensorDataPtr &tensor_data) const {
  MS_LOG(DEBUG) << "Begin, size:" << size;
  const auto stream = AscendStreamMng::GetInstance().GetStream(this->stream_id());
  auto ret = MemcpyAsync(dst, src, size, static_cast<int32_t>(kind), stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "MemcpyAsync failed!";
  }
  std::function<void(void)> callback_func = [tensor_data]() {
    // Clear tensor_data automatically.
    MS_LOG(DEBUG) << "callback_func exec, tensor_data cnt:" << tensor_data.use_count();
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ResKey res_key{GetDeviceTypeByName(device_name), device_id};
  auto res_manager = HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto callback_ret = res_manager->LaunchCallback(callback_func, this->stream_id());
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "LaunchCallback failed";
  }
}

void AscendDeviceAddress::SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind,
                                     const tensor::TensorDataPtr &tensor_data, bool sync_on_demand) const {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    MS_LOG(EXCEPTION) << "dst ptr is null, please check the address is set correctly.";
  }
  if (src == nullptr) {
    MS_LOG(EXCEPTION) << "src ptr is null, please check the address is set correctly.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  AscendHalManager::GetInstance().SetContext(device_id);

  // Only apply asynchronous copy in Pynative && ACL_MEMCPY_HOST_TO_DEVICE mode
  if (execution_mode != kPynativeMode || kind != ACL_MEMCPY_HOST_TO_DEVICE) {
    if (!sync_on_demand) {
      auto ret = SyncStreamUtils();
      if (!ret) {
        MS_LOG(EXCEPTION) << "Sync stream error!";
      }
    }
    if (!common::IsCompileSimulation()) {
      auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, dst, size, src, size, kind);
      if (ret_rt_memcpy != ACL_SUCCESS) {
        MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
      }
    }
  } else {
    if (tensor_data == nullptr) {
      // tensor_data is nullptr. Need to copy host first, then dispatch callbacks.
      SyncHostMemoryToDeviceWithCopySrc(dst, src, size, kind);
      return;
    }
    if (tensor_data->is_from_numpy()) {
      SyncHostMemoryToDeviceForTensorFromNumpy(dst, src, size, kind);
    } else {
      SyncHostMemoryToDeviceWithTensorData(dst, src, size, kind, tensor_data);
    }
  }
}

bool AscendDeviceAddress::Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src,
                                                            size_t src_size,
                                                            const tensor::TensorDataPtr &tensor_data) const {
  if (src_size / kFloat64Bytes != dst_size / kFloatBytes) {
    MS_INTERNAL_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
  SyncStreamUtils();
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src,
                                                            size_t src_size, bool sync_on_demand) const {
  if (src_size / kFloatBytes != dst_size / kFloat64Bytes) {
    MS_INTERNAL_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST, nullptr, sync_on_demand);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

void AscendDeviceAddress::SetDevicePtrDeleter() {
  if (!address_common_) {
    return;
  }

  address_common_->pointer_ref_count_->set_deleter(
    [communication_ptr = this->communication_ptr_](void *ptr, bool from_mem_pool) {
      if (ptr == nullptr || !from_mem_pool) {
        return;
      }

      if (communication_ptr != nullptr) {
        AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr);
      } else {
        AscendMemoryPool::GetInstance().FreeTensorMem(ptr);
      }
    });
}

void AscendDeviceAddress::BindDevice() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }

  // Bind device by device name and device id on the current thread.
  if (!device_name().empty()) {
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    ResKey res_key{DeviceType::kAscend, device_id};
    auto ascend_res_manager = HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(ascend_res_manager);
    if (!ascend_res_manager->BindDeviceToCurrentThread(false)) {
      MS_LOG(WARNING) << "Bind device to current thread failed.";
    }
  } else {
    MS_LOG(DEBUG) << "Device name is null.";
  }
}

void AscendDeviceAddress::SyncStream() const {
  MS_LOG(DEBUG) << "SyncStream Start!";
  auto ret = SyncStreamUtils();
  if (!ret) {
    MS_LOG(WARNING) << "Uce flag: " << UCEException::GetInstance().get_uce_flag()
                    << ", force stop flag: " << UCEException::GetInstance().get_force_stop_flag();
    if (UCEException::GetInstance().get_uce_flag()) {
      MS_LOG(EXCEPTION) << "UCEError occurs when execute.";
    } else if (UCEException::GetInstance().get_force_stop_flag()) {
      MS_LOG(EXCEPTION) << "ForceStopError occurs when execute.";
    }
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(DEBUG) << "SyncStream Finish!";
}

bool AscendDeviceAddress::SyncStream(size_t stream_id) const {
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync default stream failed.";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::CopyDeviceToHost(void *dst, const void *src, size_t size, bool async,
                                           size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, false);
}

bool AscendDeviceAddress::CopyHostToDevice(void *dst, const void *src, size_t size, bool async,
                                           size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, true);
}

bool AscendDeviceAddress::DeviceToFileDirectly(void *ptr, size_t size, const std::string &file_name,
                                               size_t stream_id) const {
  return CopyBetweenFileDeviceDirectly(ptr, file_name, size, stream_id, false);
}

bool AscendDeviceAddress::FileToDeviceDirectly(void *ptr, size_t size, const std::string &file_name,
                                               size_t stream_id) const {
  return CopyBetweenFileDeviceDirectly(ptr, file_name, size, stream_id, true);
}

bool AscendDeviceAddress::CopyBetweenFileDeviceDirectly(void *ptr, const std::string &file_name, size_t size,
                                                        size_t stream_id, bool file_to_device) const {
#if defined(RT_MEMORY_P2PDMA)
  void *dargs = AscendDmaHandle::GetInstance().GetDargs();
  void *buf = AscendDmaHandle::GetInstance().GetBuf();
  if (dargs == nullptr || buf == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(dma_lock);
  auto open_flag = file_to_device ? (O_RDWR | O_DIRECT) : (O_RDWR | O_CREAT | O_DIRECT);
  auto nvme_fd = open(file_name.c_str(), open_flag, S_IRUSR | S_IWUSR);
  if (nvme_fd < 0) {
    MS_LOG(ERROR) << "Open file failed, file name:" << file_name;
    return false;
  }
  size_t buf_size = AscendDmaHandle::GetInstance().GetSize();
  size_t count = (size + buf_size - 1) / buf_size;
  for (size_t i = 0; i < count; i++) {
    size_t ptr_offset = i * buf_size;
    size_t cur_size = (i == count - 1) ? (size - ptr_offset) : buf_size;
    if (file_to_device) {
      size_t ret_size = read(nvme_fd, buf, cur_size);
      if (ret_size != cur_size || !SyncStream(stream_id)) {
        MS_LOG(ERROR) << "Read file failed, file name:" << file_name << ", size:" << size;
        close(nvme_fd);
        return false;
      }
      DeviceToDevice(static_cast<uint8_t *>(ptr) + ptr_offset, dargs, cur_size, stream_id);
    } else {
      DeviceToDevice(dargs, static_cast<uint8_t *>(ptr) + ptr_offset, cur_size, stream_id);
      size_t ret_size = write(nvme_fd, buf, cur_size);
      if (ret_size != cur_size || !SyncStream(stream_id)) {
        MS_LOG(ERROR) << "Write file failed, file name:" << file_name << ", size:" << size;
        close(nvme_fd);
        return false;
      }
    }
  }
  close(nvme_fd);
  return true;
#else
  return false;
#endif
}

void AscendDeviceAddress::DeviceToDevice(void *dst, void *src, size_t size, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtMemcpyAsync device to device failed, the error num[" << ret << "].";
  }
  if (!AscendStreamMng::GetInstance().SyncStream(stream_id)) {
    MS_LOG(EXCEPTION) << "Sync default failed.";
  }
}

bool AscendDeviceAddress::SyncDeviceToHost(size_t size, void *const host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  SyncStream();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  CopyDeviceToHost(host_ptr, size);
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  CopyHostToDevice(host_ptr, size, nullptr);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr, bool sync_on_demand) const {
  MS_LOG(DEBUG) << "SyncDeviceToHost, Device(format:" << format() << ", type_id:" << TypeIdLabel(type_id())
                << ", size:" << GetSize() << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  if (!sync_on_demand) {
    SyncStream();
  } else {
    SyncStream(address_common_->stream_id_);
  }
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (basic_format.find(format()) != basic_format.end()) {
    if (type_id() == type) {
      CopyDeviceToHost(host_ptr, size, sync_on_demand);
      sync_ok = true;
    } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, GetDevicePtr(), GetSize(), sync_on_demand);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(GetSize());
      CopyDeviceToHost(host.data(), GetSize(), sync_on_demand);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id(), type, GetSize()};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
    }
  } else {
    if (IsOpNeedTransFormat(format())) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr, sync_on_demand);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format();
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported to trans, dev_format:" << format() << ", dev_type:" << TypeIdLabel(type_id())
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

ShapeVector AscendDeviceAddress::GetDeviceShape(ShapeVector *host_shape) const {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = GetNodeIndex();
  if (format() == kOpFormat_FRAC_NZ || format() == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format(), node_index.first, node_index.second, type_id());
  } else {
    if (!DeviceAddress::host_shape().empty()) {
      host_shape->clear();
      *host_shape = DeviceAddress::host_shape();
    }
    *host_shape = trans::PaddingShape(*host_shape, format());
    device_shape = trans::TransShapeToDevice(*host_shape, format(), node_index.first, node_index.second, type_id());
  }
  return device_shape;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr,
                                                           bool sync_on_demand) const {
  MS_LOG(DEBUG) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize()
                << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  static const std::unordered_map<mindspore::TypeId, std::string> type_id_name_map = {
    {mindspore::kNumberTypeBool, "bool"},       {mindspore::kNumberTypeInt8, "int8"},
    {mindspore::kNumberTypeInt16, "int16"},     {mindspore::kNumberTypeInt32, "int32"},
    {mindspore::kNumberTypeInt64, "int64"},     {mindspore::kNumberTypeFloat16, "float16"},
    {mindspore::kNumberTypeFloat32, "float32"}, {mindspore::kNumberTypeUInt8, "uint8"},
    {mindspore::kNumberTypeUInt16, "uint16"},   {mindspore::kNumberTypeUInt32, "uint32"},
    {mindspore::kNumberTypeUInt64, "uint64"}};
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto device_shape = GetDeviceShape(&host_shape);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  auto host_tmp = std::vector<uint8_t>(GetSize());
  CopyDeviceToHost(host_tmp.data(), GetSize(), sync_on_demand);
  auto node_index = GetNodeIndex();
  if (type_id() != type) {
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    auto host = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id(), type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr, node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDeviceImpl(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                               const void *host_ptr, const std::string &format,
                                               const tensor::TensorDataPtr &tensor_data) const {
  MS_LOG(DEBUG) << "SyncHostToDevice, Device(format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), Host(format:" << format
                << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (DeviceAddress::format() == format || basic_format.find(DeviceAddress::format()) != basic_format.end()) {
    if (type_id() == type) {
      CopyHostToDevice(host_ptr, size, tensor_data);
      sync_ok = true;
    } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(GetDevicePtr(), GetSize(), host_ptr, size, tensor_data);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id(), size};
      auto host_tmp = std::vector<uint8_t>(GetSize());
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed for device address:" << this;
        return false;
      }
      CopyHostToDevice(host_tmp.data(), GetSize(), tensor_data);
      SyncStreamUtils();
    }
  } else {
    if (IsOpNeedTransFormat(DeviceAddress::format())) {
      sync_ok = ConvertFormatAndSyncHostToDevice(shape, size, type, host_ptr, tensor_data);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << DeviceAddress::format();
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported trans, dev_format:" << DeviceAddress::format()
                  << ", dev_type:" << TypeIdLabel(type_id()) << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr, const std::string &format) const {
  return SyncHostToDeviceImpl(shape, size, type, host_ptr, format);
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type,
                                           const std::string &format, const tensor::TensorDataPtr &tensor_data) const {
  MS_EXCEPTION_IF_NULL(tensor_data);
  return SyncHostToDeviceImpl(shape, size, type, tensor_data->data(), format, tensor_data);
}

bool AscendDeviceAddress::SyncDeviceToDeviceWithDiffFormatType(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }

  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_device_address);
  BindDevice();
  auto host_shape = src_device_address->host_shape();
  if (host_shape.empty()) {
    MS_LOG(WARNING) << "Host shape of source device address is empty, emplace back shape [1],  device address size: "
                    << src_device_address->GetSize()
                    << ", device address type: " << TypeIdLabel(src_device_address->type_id());
    (void)host_shape.emplace_back(1);
  }
  auto host_tensor = std::make_shared<tensor::Tensor>(src_device_address->type_id(), host_shape);
  MS_EXCEPTION_IF_NULL(host_tensor);
  auto host_tensor_size = LongToSize(host_tensor->data().nbytes());
  auto host_tensor_type = host_tensor->data_type();
  if (!src_device_address->SyncDeviceToHost(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c())) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync device to intermediate Tensor.";
    return false;
  }
  if (!SyncHostToDevice(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c(),
                        host_tensor->device_info().host_format_)) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync intermediate tensor to device.";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (!src_device_address->MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  if (format() == src_device_address->format() && type_id() == src_device_address->type_id()) {
    return SyncDeviceToDevice(ShapeVector(), src_device_address->GetSize(), src_device_address->type_id(),
                              src_device_address->GetPtr(), src_device_address->format());
  } else {
    MS_LOG(INFO) << "Can not copy from device to device directly, format or type is different, src(format:"
                 << src_device_address->format() << ", type_id:" << TypeIdLabel(src_device_address->type_id())
                 << "), dst(format:" << format() << ", type_id:" << TypeIdLabel(type_id())
                 << ", use the intermediate Tensor copy instead.";
    return SyncDeviceToDeviceWithDiffFormatType(src_device_addr);
  }
}

bool AscendDeviceAddress::SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                                             const std::string &format) const {
  bool ret = AsyncDeviceToDevice(shape, size, type, src_ptr, format);
  if (!ret) {
    return ret;
  }
  SyncStream();
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToDevice(const DeviceAddress *src_device_addr, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  if (format() == src_device_addr->format() && type_id() == src_device_addr->type_id()) {
    return AsyncDeviceToDevice(ShapeVector(), src_device_addr->GetSize(), src_device_addr->type_id(),
                               src_device_addr->GetPtr(), src_device_addr->format(), stream_id);
  }
  MS_LOG(INFO) << "Can not copy from device to device directly, format or type is different, src(format:"
               << src_device_addr->format() << ", type_id:" << TypeIdLabel(src_device_addr->type_id())
               << "), dst(format:" << format() << ", type_id:" << TypeIdLabel(type_id())
               << ", use the intermediate Tensor copy instead.";
  return SyncDeviceToDeviceWithDiffFormatType(src_device_addr);
}

bool AscendDeviceAddress::AsyncDeviceToDevice(const ShapeVector & /* shape */, size_t size, TypeId type,
                                              const void *src_ptr, const std::string &format, size_t stream_id) const {
  MS_LOG(DEBUG) << "AsyncDeviceToDevice, dst(format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), src(format:" << format
                << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (GetDevicePtr() == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  if (GetSize() < size) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << size << ", dst size is: " << GetSize();
    return false;
  }
  if (DeviceAddress::format() != format || type_id() != type) {
    MS_LOG(ERROR) << "Format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << DeviceAddress::format() << "), type_id:" << TypeIdLabel(type_id());
    return false;
  }

  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);

  aclrtStream stream = (stream_id == SIZE_MAX) ? AscendStreamMng::GetInstance().default_stream()
                                               : AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  bool ret = MemcpyAsync(GetDevicePtr(), src_ptr, size, static_cast<int32_t>(ACL_MEMCPY_DEVICE_TO_DEVICE), stream);
  if (!ret) {
    MS_LOG(ERROR) << "MemcpyAsync failed, dst device address:" << ToString();
    return false;
  }
  return ret;
}

bool AscendDeviceAddress::AsyncHostToDevice(size_t size, TypeId type, const tensor::TensorDataPtr &tensor_data,
                                            const std::string &host_format, size_t stream_id) const {
  MS_LOG(DEBUG) << "Async host to device, size: " << size << ", host ptr: " << tensor_data->data()
                << ", device format: " << format() << ", tensor format: " << host_format
                << ", device type id: " << TypeIdToString(type_id()) << ", tensor type id: " << TypeIdToString(type)
                << ", device shape: " << GetShapeVector();
  if (format() != host_format || type_id() != type) {
    return SyncHostToDeviceImpl(GetShapeVector(), size, type, tensor_data->data(), host_format, tensor_data);
  }

  return AsyncHostToDevice(size, type, tensor_data->data(), stream_id);
}

bool AscendDeviceAddress::AsyncHostToDevice(size_t size, TypeId /* type */, const void *host_ptr,
                                            size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  MS_ERROR_IF_NULL(GetDevicePtr());

  aclrtStream stream = (stream_id == SIZE_MAX) ? AscendStreamMng::GetInstance().default_stream()
                                               : AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, GetDevicePtr(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncHostToDevice(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            const void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  MS_ERROR_IF_NULL(GetDevicePtr());
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);

  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, GetDevicePtr(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToHost(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(ERROR) << "Move data to device failed, check previous log for details.";
    return false;
  }
  MS_ERROR_IF_NULL(GetDevicePtr());
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr,
                                                           const tensor::TensorDataPtr &tensor_data) const {
  bool sync_ok = false;
  MS_LOG(DEBUG) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize()
                << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto node_index = GetNodeIndex();
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  (void)GetGroupsWithCache();
  std::vector<int64_t> device_shape;
  if (format() == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format(), node_index.first, node_index.second, type_id());
  } else {
    host_shape = trans::PaddingShape(host_shape, format());
    device_shape = trans::TransShapeToDevice(host_shape, format(), node_index.first, node_index.second, type_id());
  }
  if (type_id() != type) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id(), size};
    auto host_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    auto dst_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormat(format_args, dst_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(dst_tmp.data(), GetSize(), tensor_data);
    SyncStreamUtils();
  } else {
    const trans::FormatArgs format_args{host_ptr,   GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape, device_shape, type_id()};
    auto host_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(host_tmp.data(), GetSize(), tensor_data);
    SyncStreamUtils();
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  (void)Wait();
  if (GetDevicePtr() != nullptr && from_mem_pool()) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(GetDevicePtr());
    }
    SetDevicePtr(nullptr);
  }
}

void AscendDeviceAddress::CopyDeviceToHost(void *dst, uint64_t size, bool sync_on_demand) const {
  MS_EXCEPTION_IF_NULL(dst);
  if (hete_info_ != nullptr) {
    if (hete_info_->host_ptr_ == nullptr) {
      if (!hete_info_->file_name_.empty()) {
        MS_LOG(EXCEPTION) << "Copy from file to host is not supported yet.";
      } else {
        MS_LOG(EXCEPTION) << "Illegal heterogeneous info: empty file name and host ptr.";
      }
    }
    SyncMemory(dst, hete_info_->host_ptr_, size, ACL_MEMCPY_HOST_TO_HOST, nullptr, sync_on_demand);
  } else {
    if (GetDevicePtr() == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device ptr for device address:" << this;
    }
    SyncMemory(dst, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, nullptr, sync_on_demand);
  }
}

void AscendDeviceAddress::CopyHostToDevice(const void *src, uint64_t size,
                                           const tensor::TensorDataPtr &tensor_data) const {
  MS_EXCEPTION_IF_NULL(src);
  if (hete_info_ != nullptr) {
    if (hete_info_->host_ptr_ == nullptr) {
      if (!hete_info_->file_name_.empty()) {
        MS_LOG(EXCEPTION) << "Copy from host to file is not supported yet.";
      } else {
        MS_LOG(EXCEPTION) << "Illegal heterogeneous info: empty file name and host ptr.";
      }
    }
    SyncMemory(hete_info_->host_ptr_, src, size, ACL_MEMCPY_HOST_TO_HOST, tensor_data);
  } else {
    MS_EXCEPTION_IF_NULL(GetDevicePtr());
    if (type_id() == kObjectTypeString) {
      // NOTE: For string type, ge::StringHead.len does not include '\0', since kernel_tensor allocated size including
      // '\0', see method `CreateDeviceAddressForScalarAndString` defined in `device_address_utils.cc`, and method
      // `PrepareDataForStringValue` defined in `data_prepare_actor.cc`, so here pass `size - 1` to `head.len`.
      // NOTE: method `CopyHostToDevice` can be triggered from the two scenarios as below:
      // 1. method `CopyNoneTensorDataToDevice` in `device_address_utils.cc` passes a kernel tensor, the parameter
      // `size` include `ge::StringHead`
      // 2. method `PrepareDataForStringValue` in `data_prepare_actor.cc` passes a raw string, the parameter `size` does
      // not include `ge::StringHead`
      if (size == GetSize() && size >= sizeof(ge::StringHead)) {
        size -= sizeof(ge::StringHead);
      }
      ge::StringHead head{.addr = sizeof(ge::StringHead), .len = static_cast<int64_t>(size) - 1};
      // sync string head info from device to host
      SyncMemory(GetDevicePtr(), &head, sizeof(ge::StringHead), ACL_MEMCPY_HOST_TO_DEVICE, nullptr);
      // sync string body (real contents) from device to host
      SyncMemory(static_cast<void *>(static_cast<char *>(GetDevicePtr()) + sizeof(ge::StringHead)), src, size,
                 ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
      MS_LOG(DEBUG) << "Copy string info to device, ge::StringHead.len=" << head.len
                    << ", text=" << std::string(static_cast<const char *>(src), head.len)
                    << ", device_addr=" << GetDevicePtr();
    } else {
      SyncMemory(GetDevicePtr(), src, size, ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
    }
  }
}

bool AscendDeviceAddress::CopyBetweenHostDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id,
                                                bool host_to_device) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  auto copy_kind = host_to_device ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, copy_kind, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }
  if (async) {
    auto record_event = std::make_shared<AscendEvent>();
    record_event->set_record_stream(stream);
    record_event->RecordEvent();
    if (loadable_mem_ == nullptr) {
      loadable_mem_ = std::make_unique<LoadableMember>();
    }
    loadable_mem_->swap_event_.device_event_ = record_event;
  } else {
    if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  return true;
}

bool AscendDeviceAddress::CopyDeviceToHost(void *dst, const void *src, const size_t &size) const {
  SyncMemory(dst, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendDeviceAddress::CopyHostToDevice(void *dst, const void *src, const size_t &size) const {
  SyncMemory(dst, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToHost(size_t size, void *host_ptr, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  BindDevice();
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  aclrtStream stream = nullptr;
  if (stream_id == SIZE_MAX) {
    auto cur_stream_id = AscendStreamMng::GetInstance().current_stream();
    stream = AscendStreamMng::GetInstance().GetStream(cur_stream_id);
    if (stream == nullptr) {
      stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
    }
  } else {
    stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  }
  MS_ERROR_IF_NULL(stream);
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncHostToDevice(size_t size, const void *host_ptr, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  BindDevice();
  auto cur_stream_id = AscendStreamMng::GetInstance().current_stream();
  auto stream = AscendStreamMng::GetInstance().GetStream(cur_stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
    cur_stream_id = kDefaultStreamIndex;
  }
  MS_ERROR_IF_NULL(stream);
  SyncHostMemoryToDeviceWithCopySrc(GetDevicePtr(), host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_id);
  return true;
}

AscendDeviceAddress::~AscendDeviceAddress() {
  try {
    // Only release offload memory, release device memory when `kernel_tensor_` in base class destroyed, because maybe
    // multi GPUDeviceAddress objects use same device pointer in ref case.
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    (void)Wait();
    LoadableDeviceAddress::ReleaseResource();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed.";
  }
}

int64_t AscendDeviceAddress::GetGroupsWithCache() const {
  auto node = GetNodeIndex();
  if (node.first != nullptr) {
    groups_ = common::AnfAlgo::GetAttrGroups(node.first, node.second);
  }
  return groups_;
}

bool AscendDeviceAddress::CopyDeviceToHostWithoutSyncStream(void *dst, size_t dst_size, const void *src,
                                                            size_t src_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  auto ret = CALL_ASCEND_API(aclrtMemcpy, dst, dst_size, src, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "AclrtMemcpy failed, error code: " << ret;
  }
  return (ret != ACL_SUCCESS);
}

DeviceAddressPtr AscendDeviceAddress::CloneDeviceAddress() {
  auto clone_device_address = std::make_shared<AscendDeviceAddress>();
  DeviceAddress::CloneDeviceAddress(clone_device_address);
  clone_device_address->set_communication_ptr(communication_ptr_);
  return clone_device_address;
}

/*
 * Feature group: Dump
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Load tensor to host and create tensor_data object for the loaded tensor.
 */
mindspore::tensor::TensorPtr AscendDeviceAddress::LoadMemToHost(const std::string &tensor_name,
                                                                const ShapeVector &host_shape, TypeId host_type,
                                                                bool trans_flag, bool async_copy) const {
  ShapeVector corrected_host_shape = host_shape;
  if (host_type == kNumberTypeInt4 && !corrected_host_shape.empty()) {
    constexpr int64_t kNumber2 = 2;
    corrected_host_shape.back() *= kNumber2;
  }
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, corrected_host_shape);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = LongToSize(out_tensor->data().nbytes());
  if (host_size == 0) {
    MS_LOG(INFO) << "Tensor size is 0 for tensor: " << tensor_name;
    return std::make_shared<mindspore::tensor::Tensor>();
  }
  if (host_type == kNumberTypeInt4) {
    const int int4_nums_per_byte = 2;
    host_size = out_tensor->DataSize() / int4_nums_per_byte;
  }
  bool ret_sync = false;
  if (async_copy) {
    if (trans_flag) {
      ret_sync = SyncDeviceToHost(corrected_host_shape, host_size, host_type, out_tensor->data_c());
    } else {
      ret_sync = SyncDeviceToHost(host_size, out_tensor->data_c());
    }
  } else {
    // copy device to host using sync mode
    auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, out_tensor->data_c(), host_size, GetDevicePtr(), GetSize(),
                                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_LOG(ERROR) << "SyncDeviceToHost: aclrtMemcpy mem size[" << GetSize() << "] fail, ret[" << ret_rt_memcpy << "]";
      return nullptr;
    } else {
      ret_sync = true;
    }
  }
  if (!ret_sync) {
    MS_LOG(ERROR) << "Convert format or Copy device mem to host failed";
    return nullptr;
  }
  return out_tensor;
}

bool BindDeviceToCurrentThread() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(void *host_ptr, const void *device_ptr, size_t size,
                                           const std::string &device_name, uint32_t device_id, mindspore::Format format,
                                           const ShapeVector &shape, size_t stream_id,
                                           const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  MS_EXCEPTION_IF_NULL(device_ptr);
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);

  if (!BindDeviceToCurrentThread()) {
    MS_LOG(WARNING) << "Bind device to current thread failed.";
  }

  if (stream_id != kDefaultStreamIndex) {
    if (!AscendStreamMng::GetInstance().SyncStream(kDefaultStreamIndex)) {
      MS_LOG(ERROR) << "Sync stream failed, stream id: " << kDefaultStreamIndex;
      return false;
    }
  }

  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }

  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream failed, stream id: " << stream_id;
    return false;
  }
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(void *device_ptr, const void *host_ptr, size_t size,
                                           const std::string &device_name, uint32_t device_id, mindspore::Format format,
                                           const ShapeVector &shape, size_t stream_id,
                                           const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(device_ptr);
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);

  if (!BindDeviceToCurrentThread()) {
    MS_LOG(WARNING) << "Bind device to current thread failed.";
  }

  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }

  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream failed, stream id: " << stream_id;
    return false;
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
