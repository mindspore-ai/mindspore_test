/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/hccl/hcom_all_to_all.h"
#include <algorithm>
#include "plugin/res_manager/ascend/hccl_adapter/hccl_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"

namespace mindspore::kernel {
bool HcomAllToAllKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcclAllToAll launch";
  if (inputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid AllToAll input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto shape = inputs[0]->GetShapeVector();
  uint64_t count = SizeOf(shape) / rank_size_;
  params_.recvcount = count;
  params_.sendcount = count;

  auto output_device_ptr = outputs.empty() ? nullptr : outputs[0]->device_ptr();
  if (NeedReGetHcom()) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need re-get hcom";
    comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  }
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllToAll(inputs[0]->device_ptr(), output_device_ptr, params_,
                                                                   data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllToAll failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool HcomAllToAllKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!HcclKernel::Init(inputs, outputs)) {
    return false;
  }

  if (hccl_data_type_list_.empty()) {
    auto recv_type = GetValue<TypePtr>(primitive_->GetAttr(kAttrRecvType));
    MS_EXCEPTION_IF_NULL(recv_type);
    data_type_ = HcomUtil::ConvertHcclType(recv_type->type_id());
  } else {
    data_type_ = hccl_data_type_list_[0];
  }

  if (!CommManager::GetInstance().GetRankSize(group_, &rank_size_) || rank_size_ == 0) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group_ << " failed.";
  }

  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "AlltoAll should have at least one input, but got 0 input.";
  }

  return true;
}

MS_HCCL_REG_KERNEL(AllToAll, HcomAllToAllKernel);
}  // namespace mindspore::kernel
