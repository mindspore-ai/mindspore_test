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

#include "kernel/ascend/pyboost/customize/dist_comm_gather_into_tensor.h"

#include <memory>
#include <string>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/pyboost/customize/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommGatherIntoTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &other_tensor,
                                             const BaseTensorPtr &input_tensor, const Int64ImmPtr &rank_size,
                                             const Int64ImmPtr &dst, const Int64ImmPtr &rank_id,
                                             const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, rank_size, dst, rank_id, group);
  auto dst_rank = GetValue<int64_t>(dst);
  auto local_rank = GetValue<int64_t>(rank_id);
  if (local_rank == dst_rank) {
    PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
    op->set_outputs({other_tensor});
  } else {
    PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
    PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());
  }

  auto run_func = [op, other_tensor, input_tensor, local_rank, dst_rank, group]() {
    if (local_rank == dst_rank) {
      PyBoostUtils::MallocOpInputs(op->device_context(), other_tensor, input_tensor);
    } else {
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
    }

    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);
    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type](const HcclComm &hccl_comm,
                                                                                     void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(input_data_ptr, output_data_ptr, hccl_count,
                                                                        hccl_data_type, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcomGather failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
