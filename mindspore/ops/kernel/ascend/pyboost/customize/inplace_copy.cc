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

#include "kernel/ascend/pyboost/customize/inplace_copy.h"
#include <algorithm>
#include <memory>
#include <string>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/auto_generate/inplace_copy.h"
#include "runtime/device/res_manager/utils/utils.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_address.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
device::DeviceType GetTensorDeviceType(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor,
                                       const std::string &name) {
  auto device_type = device::DeviceType::kCPU;
  auto addr = tensor->device_address();
  if (addr != nullptr) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(addr);
    device_type = device_address->GetDeviceType();
  } else {
    MS_LOG(DEBUG) << "For InplaceCopy, " << name << " don't have device_address, set it to host tensor.";
  }
  if (MS_UNLIKELY(device_type != device::DeviceType::kAscend && device_type != device::DeviceType::kCPU)) {
    MS_LOG(EXCEPTION) << "For InplaceCopy, device_type must be Ascend or CPU, but got "
                      << GetDeviceNameByType(device_type);
  }
  return device_type;
}

void *GetDevicePtrFromTensor(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);

  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  if (tensor->storage_offset() == 0) {
    return device_address->GetMutablePtr();
  }

  size_t offset = mindspore::abstract::TypeIdSize(tensor->data_type()) * tensor->storage_offset();
  return reinterpret_cast<char *>(device_address->GetMutablePtr()) + offset;
}

bool IsComplexTensor(const TensorPtr &tensor) {
  return tensor->data_type() == kNumberTypeComplex || tensor->data_type() == kNumberTypeComplex64 ||
         tensor->data_type() == kNumberTypeComplex128;
}
}  // namespace

tensor::TensorPtr InplaceCopyD2D(const std::shared_ptr<OpRunner> &op, const TensorPtr &dst, const TensorPtr &src) {
  MS_LOG(DEBUG) << "Call InplaceCopy D2D start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dst, src);
  op->set_outputs({dst});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, dst, src]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, dst, src);

    // Inplace output need be front
    LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), dst, src);
    MS_LOG(DEBUG) << "Launch InplaceCopy D2D end";
  }));
  return op->output(0);
}

tensor::TensorPtr InplaceCopyH2D(const std::shared_ptr<OpRunner> &op, const TensorPtr &dst, const TensorPtr &src,
                                 const bool &non_blocking) {
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  auto dst_device_type = GetTensorDeviceType(op, dst, "dst");
  auto src_device_type = GetTensorDeviceType(op, src, "src");
  if (dst_device_type != device::DeviceType::kAscend || src_device_type != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "For InplaceCopyH2D, dst must be device tensor and src must be host tensor. But got dst on "
                      << GetDeviceNameByType(dst_device_type) << " and src on " << GetDeviceNameByType(src_device_type);
  }

  auto src_storage_offset = LongToSize(src->storage_offset());
  if (IsComplexTensor(src) || src_storage_offset != 0 || src->Size() != dst->Size() || src->data_c() == nullptr) {
    MS_LOG(DEBUG) << "InplaceCopyH2D don't support complex or discontiguous src yet.";
    return InplaceCopyD2D(op, dst, src);
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dst);
  dst->set_sync_status(kNeedSyncDeviceToHost);
  op->set_outputs({dst});

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, dst, src, non_blocking]() {
    auto device_context = op->device_context();
    auto stream_id = op->stream_id();

    PyBoostUtils::MallocOpInputs(device_context, dst);

    if (src->Size() > 0 && !common::IsCompileSimulation()) {
      runtime::OpExecutor::DispatchLaunchTask([device_context, stream_id, dst, src, non_blocking]() {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyNativeLaunchTask, "InplaceCopyH2D", false);
        device_context->device_res_manager_->BindDeviceToCurrentThread(false);
        void *dst_ptr = GetDevicePtrFromTensor(dst);
        void *src_ptr = src->data_c();

        if (MS_UNLIKELY(dst_ptr == nullptr)) {
          MS_LOG(ERROR) << "dst device_ptr: " << dst_ptr << ", Maybe you free the device memory before InplaceCopyH2D"
                        << ", Check if dst.storage().resize_(0) is used before copy_.";
        }

        if (!non_blocking) {
          if (!device_context->device_res_manager_->SyncStream(stream_id)) {
            MS_LOG(EXCEPTION) << "For InplaceCopyH2D, SyncStream failed with non_blocking = " << non_blocking;
          }
          auto ret_rt_memcpy =
            CALL_ASCEND_API(aclrtMemcpy, dst_ptr, dst->Size(), src_ptr, src->Size(), ACL_MEMCPY_HOST_TO_DEVICE);
          if (ret_rt_memcpy != ACL_SUCCESS) {
            MS_LOG(EXCEPTION) << "For InplaceCopyH2D, aclrtMemcpy call failed with error = " << ret_rt_memcpy
                              << ", src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr << ", copySize: " << src->Size();
          }
          MS_LOG(DEBUG) << "Launch InplaceCopyH2D SyncCopy end";
        } else {
          // Use temp buffer to ensure host data is not modified during async copying
          std::shared_ptr<uint8_t[]> host_buffer(new (std::nothrow) uint8_t[src->Size()]);
          MS_EXCEPTION_IF_NULL(host_buffer);
          auto ret = memcpy_s(host_buffer.get(), src->Size(), src_ptr, src->Size());
          if (ret == ERANGE) {
            device::ConvertSameType(host_buffer.get(), src_ptr, src->Size(), src->data_type());
          }

          // call aclrtMemcpyAsync to copy host tor device async
          auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
          auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpyAsync, dst_ptr, dst->Size(), src_ptr, src->Size(),
                                               ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
          if (ret_rt_memcpy != ACL_SUCCESS) {
            MS_LOG(EXCEPTION) << "For InplaceCopyH2D, aclrtMemcpyAsync call failed with error = " << ret_rt_memcpy
                              << ", src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr << ", copySize: " << src->Size();
          }

          std::function<void(void)> callback_func = [host_buffer]() {
            // Clear host_buffer automatically.
            MS_LOG(DEBUG) << "InplaceCopyH2D callback_func exec, host_buffer cnt:" << host_buffer.use_count();
          };

          auto ms_context = MsContext::GetInstance();
          MS_EXCEPTION_IF_NULL(ms_context);
          auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
          auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(
            device::ResKey{device::DeviceType::kAscend, device_id});
          MS_EXCEPTION_IF_NULL(res_manager);
          auto callback_ret = res_manager->LaunchCallback(callback_func, stream_id);
          if (!callback_ret) {
            MS_LOG(EXCEPTION) << "InplaceCopyH2D LaunchCallback failed on device " << device_id;
          }

          auto sync_mode = runtime::RuntimeConf::GetInstance()->launch_blocking();
          if (sync_mode) {
            if (!device_context->device_res_manager_->SyncStream(stream_id)) {
              MS_LOG(EXCEPTION) << "SyncStream failed for InplaceCopyH2D AsyncCopy.";
            }
          }
          MS_LOG(DEBUG) << "Launch InplaceCopyH2D AsyncCopy end";
        }
      });
    }
  }));

  return op->output(0);
}

tensor::TensorPtr InplaceCopyD2H(const std::shared_ptr<OpRunner> &op, const TensorPtr &dst, const TensorPtr &src,
                                 const bool &non_blocking) {
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  auto dst_device_type = GetTensorDeviceType(op, dst, "dst");
  auto src_device_type = GetTensorDeviceType(op, src, "src");
  if (dst_device_type != device::DeviceType::kCPU || src_device_type != device::DeviceType::kAscend) {
    MS_LOG(EXCEPTION) << "For InplaceCopyD2H, dst must be host tensor and src must be device tensor. But got dst on "
                      << GetDeviceNameByType(dst_device_type) << " and src on " << GetDeviceNameByType(src_device_type);
  }

  auto dst_storage_offset = LongToSize(dst->storage_offset());
  if (dst_storage_offset != 0 || src->Size() != dst->Size() || dst->data_c() == nullptr) {
    MS_LOG(DEBUG) << "InplaceCopyD2H don't support discontiguous dst yet.";
    return InplaceCopyD2D(op, dst, src);
  }

  auto dst_addr = dst->device_address();
  if (dst_addr != nullptr) {
    dst->data_sync();
    dst->set_device_address(nullptr);
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), src);
  dst->set_sync_status(kNeedSyncHostToDevice);
  op->set_outputs({dst});

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, dst, src, non_blocking]() {
    auto device_context = op->device_context();
    auto stream_id = op->stream_id();

    PyBoostUtils::MallocOpInputs(device_context, src);

    if (src->Size() > 0 && !common::IsCompileSimulation()) {
      runtime::OpExecutor::DispatchLaunchTask([device_context, stream_id, dst, src, non_blocking]() {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyNativeLaunchTask, "InplaceCopyD2H", false);
        device_context->device_res_manager_->BindDeviceToCurrentThread(false);
        void *dst_ptr = dst->data_c();
        void *src_ptr = GetDevicePtrFromTensor(src);

        if (MS_UNLIKELY(src_ptr == nullptr)) {
          MS_LOG(ERROR) << "src device_ptr: " << src_ptr << ", Maybe you free the device memory before InplaceCopyD2H"
                        << ", Check if src.storage().resize_(0) is used before copy_.";
        }

        if (!non_blocking) {
          if (!device_context->device_res_manager_->SyncStream(stream_id)) {
            MS_LOG(EXCEPTION) << "For InplaceCopyD2H, SyncStream failed with non_blocking = " << non_blocking;
          }
          auto ret_rt_memcpy =
            CALL_ASCEND_API(aclrtMemcpy, dst_ptr, dst->Size(), src_ptr, src->Size(), ACL_MEMCPY_DEVICE_TO_HOST);
          if (ret_rt_memcpy != ACL_SUCCESS) {
            MS_LOG(EXCEPTION) << "For InplaceCopyD2H, aclrtMemcpy call failed with error = " << ret_rt_memcpy
                              << ", src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr << ", copySize: " << src->Size();
          }
          MS_LOG(DEBUG) << "Launch InplaceCopyD2H SyncCopy end";
        } else {
          auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
          auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpyAsync, dst_ptr, dst->Size(), src_ptr, src->Size(),
                                               ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
          if (ret_rt_memcpy != ACL_SUCCESS) {
            MS_LOG(EXCEPTION) << "For InplaceCopyD2H, aclrtMemcpyAsync call failed with error = " << ret_rt_memcpy
                              << ", src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr << ", copySize: " << src->Size();
          }
          auto sync_mode = runtime::RuntimeConf::GetInstance()->launch_blocking();
          if (sync_mode) {
            if (!device_context->device_res_manager_->SyncStream(stream_id)) {
              MS_LOG(EXCEPTION) << "SyncStream failed for InplaceCopyD2H AsyncCopy.";
            }
          }
          MS_LOG(DEBUG) << "Launch InplaceCopyD2H AsyncCopy end";
        }
      });
    }
  }));

  return op->output(0);
}

tensor::TensorPtr InplaceCopyH2H(const std::shared_ptr<OpRunner> &op, const TensorPtr &dst, const TensorPtr &src) {
  if (dst->shape() == src->shape() && dst->Dtype()->type_id() == src->Dtype()->type_id()) {
    constexpr size_t kGrainSize = 32768;
    auto copy_size = std::max(dst->DataSize(), src->DataSize());
    if (copy_size < kGrainSize) {
      auto dst_addr = dst->device_address();
      if (dst_addr != nullptr) {
        dst->data_sync();
        dst->set_device_address(nullptr);
      }
      auto src_addr = src->device_address();
      if (src_addr != nullptr) {
        src->data_sync();
        src->set_device_address(nullptr);
      }
      auto size = dst->Size();
      if (size > 0 && !common::IsCompileSimulation()) {
        auto ret = memcpy_s(dst->data_c(), size, src->data_c(), size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "InplaceCopyH2H fast copy failed, memcpy_s failed with error: " << ret;
        }
      }
      op->set_outputs({dst});
      return op->output(0);
    }
  }

  auto cpu_copy_op = CREATE_PYBOOST_OP(InplaceCopy, "CPU");
  (void)cpu_copy_op->Call(dst, src, std::make_shared<BoolImm>(false));
  op->set_outputs(cpu_copy_op->outputs());
  return op->output(0);
}

tensor::TensorPtr InplaceCopyAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dst,
                                             const TensorPtr &src, const BoolImmPtr &non_blocking) {
  auto is_contiguous = src->is_contiguous() && dst->is_contiguous();
  if (dst->shape() != src->shape() || dst->data_type() != src->data_type() || !is_contiguous) {
    MS_LOG(DEBUG) << "InplaceCopy H2D/D2H/H2H don't support BroadCast, DtypeCast, empty and discontiguous src/dst yet.";
    return InplaceCopyD2D(op, dst, src);
  }

  auto dst_device_type = GetTensorDeviceType(op, dst, "dst");
  auto src_device_type = GetTensorDeviceType(op, src, "src");
  MS_LOG(DEBUG) << "InplaceCopy Launch with dst tensor on " << GetDeviceNameByType(dst_device_type)
                << " and src tensor on " << GetDeviceNameByType(src_device_type)
                << " with non_blocking=" << non_blocking->value();

  if (dst_device_type == device::DeviceType::kAscend) {
    if (src_device_type == device::DeviceType::kAscend) {
      return InplaceCopyD2D(op, dst, src);
    } else {
      return InplaceCopyH2D(op, dst, src, non_blocking->value());
    }
  } else if (dst_device_type == device::DeviceType::kCPU) {
    if (src_device_type == device::DeviceType::kAscend) {
      return InplaceCopyD2H(op, dst, src, non_blocking->value());
    } else {
      return InplaceCopyH2H(op, dst, src);
    }
  }

  return InplaceCopyD2D(op, dst, src);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
