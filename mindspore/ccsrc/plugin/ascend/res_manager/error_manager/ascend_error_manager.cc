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

#include "plugin/ascend/res_manager/error_manager/ascend_error_manager.h"

#include <memory>
#include <mutex>
#include <vector>
#include "include/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "ir/device_type.h"
#include "ir/tensor_new.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "tools/error_handler/error_handler.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "tools/error_handler/error_config.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace tools {
namespace ascend {
namespace {
SNAPSHOT_MANAGER_REG(kAscendDevice, AscendSnapshotMgr);

inline ErrorType GetErrorType(int error_code) {
  switch (error_code) {
    case ACL_ERROR_RT_DEVICE_MEM_ERROR:
      return ErrorType::kDeviceMemError;
    case ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR:
      return ErrorType::kHbmMultBitEccError;
    case ACL_ERROR_RT_COMM_OP_RETRY_FAIL:
      return ErrorType::kCommOpRetryFailError;
    case ACL_ERROR_RT_DEVICE_TASK_ABORT:
      return ErrorType::kForceStopError;
    case ACL_ERROR_RT_SUSPECT_REMOTE_ERROR:
      return ErrorType::kSuspectRemoteError;
    default:
      return ErrorType::kUnknownError;
  }
}

void RunFailCallback(const char *caller_file, int caller_line, const char *caller_name, const std::string &api_info,
                     bool throw_exception) {
  auto aclrt_get_last_error = mindspore::device::ascend::aclrtGetLastError_;
  auto acl_get_recent_err_msg = mindspore::device::ascend::aclGetRecentErrMsg_;
  if (aclrt_get_last_error != nullptr && (mindspore::tools::TftConfig::GetInstance()->IsEnableUCE() ||
                                          mindspore::tools::TftConfig::GetInstance()->IsEnableHCCE())) {
    auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
    auto error_type = GetErrorType(error_code);
    mindspore::tools::ErrorHandler::GetInstance().ProcessError(
      mindspore::tools::FuncInfo{caller_file, caller_line, caller_name, api_info}, error_code, acl_get_recent_err_msg,
      error_type, throw_exception);
  }
  if (mindspore::tools::TftConfig::GetInstance()->IsEnableARF()) {
    if (aclrt_get_last_error != nullptr) {
      auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
      MS_LOG(DEBUG) << "Call ascend api <" << api_info << "> in <" << caller_name << "> at " << caller_file << ":"
                    << caller_line << " failed, error code [" << error_code << "].";
      if (error_code == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
        mindspore::tools::ErrorHandler::GetInstance().SetForceStopFlag(true);
      }
    }
  }
}

REGISTER_COMMON_CALLBACK(RunFailCallback);
}  // namespace

AscendSnapshotMgrPtr AscendSnapshotMgr::GetInstance() {
  auto instance = SnapshotMgr::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(instance);
  AscendSnapshotMgrPtr ptr_inst = std::dynamic_pointer_cast<AscendSnapshotMgr>(instance);
  if (ptr_inst->async_copy_event_ != nullptr) {
    return ptr_inst;
  }

  static std::mutex mtx;
  std::lock_guard<std::mutex> gurad(mtx);
  if (ptr_inst->async_copy_event_ == nullptr) {
    if (CALL_ASCEND_API(aclrtCreateEventExWithFlag, &ptr_inst->async_copy_event_, ACL_EVENT_SYNC) != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Create async event failed";
    }
  }

  return ptr_inst;
}

void AscendSnapshotMgr::Clear() {
  Reset();
  if (async_copy_event_ != nullptr) {
    auto ret = CALL_ASCEND_API(aclrtDestroyEvent, async_copy_event_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtDestroyEvent failed with return value " << ret;
    }
    async_copy_event_ = nullptr;
  }
}

AscendSnapshotMgr::~AscendSnapshotMgr() { Clear(); }

void AscendSnapshotMgr::RecordEvent(aclrtStream stream) {
  aclError ret = CALL_ASCEND_API(aclrtRecordEvent, async_copy_event_, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtRecordEvent failed, error code is " << ret;
  }
}

void AscendSnapshotMgr::StreamWaitEvent(aclrtStream stream) {
  aclError ret = CALL_ASCEND_API(aclrtStreamWaitEvent, stream, async_copy_event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtStreamWaitEvent failed, error code is " << ret;
  }
}

void AscendSnapshotMgr::SaveParameters(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  int index = 0;
  for (const auto &node : weights) {
    index += 1;
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto out_addr = session::AnfRuntimeAlgorithm::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto iter = saved_params_.find(param->name());
      if (iter == saved_params_.end()) {
        MS_LOG(WARNING) << "Can not find parameter " << param->name() << " in saved parameters.";
        continue;
      }
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (iter->second == nullptr) {
        // NOTE: here use `pin_mem_allocator` to allocate host memory for saving snapshot, otherwise there would be more
        // overhead when copying data from device to host
        const auto &shape = tensor->shape();
        const auto &dtype = tensor->data_type();
        auto device_address = DeviceAddressMaker(nullptr, dtype, shape)
                                .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                                .make_device_address();

        auto ascend_device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(
          device::GetDeviceNameByType(device::DeviceType::kAscend));
        if (ascend_device_ctx == nullptr || ascend_device_ctx->device_res_manager_ == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot find Ascend device context. ascend_device_ctx or device_res_manager is null.";
        }

        auto pin_memory_allocator = ascend_device_ctx->device_res_manager_->pin_mem_allocator();
        std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->set_allocator(pin_memory_allocator);

        auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(
          device::GetDeviceNameByType(device::DeviceType::kCPU));
        bool allocate_mem_ret = device_ctx->device_res_manager_->AllocateMemory(
          std::dynamic_pointer_cast<device::DeviceAddress>(device_address).get());
        if (!allocate_mem_ret) {
          MS_LOG(EXCEPTION) << "Tensor.pin_memory allocate memory failed!";
        }

        auto tensor_ptr = std::make_shared<tensor::Tensor>(dtype, shape, device_address);
        saved_params_[param->name()] = tensor_ptr;
      }
      auto host_tensor = saved_params_[param->name()];
      auto size = tensor->Size();
      MS_LOG(INFO) << "Copy parameter " << param->name() << " with size " << size << " " << index << "/"
                   << weights.size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
    }
  }
}

bool NeedSaveAsyncCkpt() {
  static bool disable_ckpt_d2h_async = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC") != "1";
  if (MS_LIKELY(disable_ckpt_d2h_async)) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_NEED_CKPT)) {
    return false;
  }

  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
  auto checkpoint_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", checkpoint_steps: " << checkpoint_steps
                << ", last_triggered_step:" << last_triggered_step;
  return cur_step >= (last_triggered_step + checkpoint_steps);
}

bool NeedSaveSnapshot() {
  if (!TftConfig::IsEnableStepTRE()) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto last_save_step = AscendSnapshotMgr::GetInstance()->LastSaveStep();
  auto snapshot_steps = TftConfig::GetSnapShotSteps();
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", snapshot_steps: " << snapshot_steps
                << ", last_save_step:" << last_save_step;
  return last_save_step > 0 ? cur_step >= (last_save_step + snapshot_steps) : cur_step > snapshot_steps;
}
}  // namespace ascend
}  // namespace tools
}  // namespace mindspore
