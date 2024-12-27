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

#include "kernel/ascend/pyboost/customize/dist_comm_scatter.h"

#include <memory>
#include <string>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/pyboost/customize/comm_common.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &other_tensor,
                                    const ValueTuplePtr &scatter_list, const Int64ImmPtr &rank_size,
                                    const Int64ImmPtr &src, const Int64ImmPtr &rank_id, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, scatter_list, rank_size, src, rank_id, group);
  op->set_outputs({other_tensor});

  auto src_rank = GetValue<int64_t>(src);
  auto local_rank = GetValue<int64_t>(rank_id);
  std::vector<BaseTensorPtr> scatter_tensors = ConvertValueTupleToVector<BaseTensorPtr>(scatter_list);
  auto rank_size_imm = GetValue<int64_t>(rank_size);
  auto input_shape = other_tensor->shape();
  input_shape[0] = static_cast<int64_t>(input_shape[0] * rank_size_imm);
  BaseTensorPtr input_tensor =
    std::make_shared<tensor::Tensor>(static_cast<TypeId>(other_tensor->data_type_c()), input_shape);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, scatter_tensors, input_tensor);

  auto run_func = [op, other_tensor, input_tensor, local_rank, src_rank, group, rank_size_imm, scatter_tensors]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputs(device_context, other_tensor, input_tensor, scatter_tensors);

    auto [hccl_count, hccl_data_type] =
      HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor, rank_size_imm);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto size = scatter_tensors[0]->Size();
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, src_rank, local_rank, size,
                        scatter_tensors, op_name, rank_size_imm](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      if (local_rank == src_rank) {
        for (int r = 0; r < rank_size_imm; r++) {
          uint64_t offset = (uint64_t)(r * size);
          auto data_ptr = GetDevicePtrFromTensor(op_name, scatter_tensors[r]);
          auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, static_cast<char *>(input_data_ptr) + offset, size, data_ptr,
                                        size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
          }
        }
      }

      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclScatter(
        input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, src_rank, hccl_comm, comm_stream_ptr);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclScatter failed, ret:" << hccl_result;
      }
    };

    auto post_func = [input_tensor, scatter_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor,
                                                                      op->output(0), scatter_tensors);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
