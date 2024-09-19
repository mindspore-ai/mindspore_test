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

#include "kernel/ascend/pyboost/customize/inner_comm_all_reduce.h"

#include <memory>
#include <string>
#include "kernel/common/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/pyboost/customize/comm_common.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommAllReduceAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                       const StringImmPtr &op_type, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, op_type, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, input_tensor, op_type, group]() {
    PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
    PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());

    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);
    auto op_type_enum = HcomUtil::GetHcomReduceOpType(GetValue<std::string>(op_type));

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum](
                         const HcclComm &hccl_comm, void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllReduce(
        input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcomRecv failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
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
