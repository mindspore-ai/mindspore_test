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

#include "kernel/ascend/pyboost/customize/inner_comm_isend.h"

#include <memory>
#include <string>
#include "kernel/common/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/pyboost/customize/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommIsendAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                   const Int64ImmPtr &dst, const StringImmPtr &group, const Int64ImmPtr &tag) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);

  op->set_outputs({input_tensor});
  op->CreateOutputSimpleInfoForView();

  auto run_func = [op, input_tensor, dst, group]() {
    auto device_context = op->device_context();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);

    auto dst_imm = GetValue<int64_t>(dst);

    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);
    auto input_data_ptr = GetDevicePtrFromTensor(op->primitive()->name(), input_tensor);
    auto launch_func = [input_data_ptr, hccl_count, hccl_data_type, dst_imm](const HcclComm &hccl_comm,
                                                                             void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclSend(input_data_ptr, hccl_count, hccl_data_type, dst_imm,
                                                                   comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcomRecv failed, ret:" << hccl_result;
      }
    };

    auto post_func = [device_context, input_tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), device_context,
                                                                      comm_stream_id, event, input_tensor);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };

  if (runtime::OpExecutor::NeedSync()) {
    run_func();
  } else {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(std::make_shared<runtime::PassthroughDeviceTask>(run_func));
  }
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
