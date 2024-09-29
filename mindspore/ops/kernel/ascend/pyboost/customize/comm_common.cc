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

#include "kernel/ascend/pyboost/customize/comm_common.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/comm_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void CommonCommAscendFunc(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                          const StringImmPtr &group, const std::function<void(const HcclComm &, void *)> &launch_func,
                          const std::function<void(const DeviceEventPtr &, size_t)> &post_func) {
  const auto &op_name = op->primitive()->name();
  MS_LOG(DEBUG) << "Run device task " << op_name << " end";

  runtime::Pipeline::Get().launch_stage()->Wait();

  const auto &group_str = GetValue<std::string>(group);
  const auto &hccl_comm = device::ascend::AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_str);

  auto comm_handle = op->comm_handle();
  auto device_context = op->device_context();
  static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) ||
                     MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON";

  // Need to bind context if the comm_op is the first op launched in this thread.
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto comm_stream_id = device_context->device_res_manager_->GetCommunicationStreamID();

  [device_context, op_stream_id = op->stream_id(), comm_handle, hccl_comm, comm_stream_id, op_name, launch_func]() {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask,
                                       op_name, false);

    CommUtils::GetInstance().SyncOpStream(device_context, op_stream_id, comm_stream_id);
    auto comm_stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(comm_stream_id);

    if (launch_func) {
      launch_func(hccl_comm, comm_stream_ptr);
      if (sync) {
        if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
          MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name;
        }
      }
    }
    comm_handle->RecordEvent(comm_stream_id);
  }();

  if (post_func) {
    post_func(comm_handle->event(), comm_stream_id);
  } else {
    // Default post function
    runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(
      op->primitive()->name(), device_context, comm_stream_id, comm_handle->event(), input_tensor, op->output(0));
  }
  comm_handle->UpdateTaskId(comm_stream_id);

  if (sync) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name;
    }
  }
  MS_LOG(DEBUG) << "Run device task " << op_name << " end";
}

void *GetDevicePtrFromTensor(const std::string &op_name, const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);

  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  auto storage_info = tensor->storage_info();
  if (storage_info == nullptr) {
    return device_address->GetMutablePtr();
  }

  if (!storage_info->is_contiguous) {
    MS_EXCEPTION(ValueError) << op_name
                             << " does not support not-contiguous tensor. Please call tensor.contiguous() firstly.";
  }

  if (storage_info->storage_offset == 0) {
    return device_address->GetMutablePtr();
  }

  size_t offset = mindspore::abstract::TypeIdSize(tensor->data_type()) * storage_info->storage_offset;
  // tensor is contiguous, add offset for addr
  return reinterpret_cast<char *>(device_address->GetMutablePtr()) + offset;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
