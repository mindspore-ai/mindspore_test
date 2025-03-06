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

#include "plugin/device/ascend/kernel/hccl/pyboost/inner_comm_all_gather.h"

#include <memory>
#include <string>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/kernel/hccl/pyboost/comm_common.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommAllGatherAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                       const Int64ImmPtr &rank_size, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, rank_size, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, input_tensor, group]() {
    PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
    PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());

    auto hccl_count = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor).first;

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto input_dtype = input_tensor->data_type();
    std::string group_name = GetValue<std::string>(group);
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, input_dtype, group_name](const HcclComm &,
                                                                                              void *comm_stream_ptr) {
      auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
      auto comm_result =
        comm_lib->AllGather(input_data_ptr, output_data_ptr, hccl_count, input_dtype, group_name, comm_stream_ptr);
      if (!comm_result) {
        MS_LOG(EXCEPTION) << "InnerCommAllGather failed, ret:" << comm_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
