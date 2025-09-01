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

#include "kernel/ascend/hccl/hcom_all_to_all_v.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace kernel {
bool HcomAlltoAllVKernel::GetAllToAllVParam(const std::vector<int64_t> &send_numel_list,
                                            const std::vector<int64_t> &recv_numel_list) {
  auto send_offset_list = primitive_->HasAttr(kAttrSendOffsetList)
                            ? GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrSendOffsetList))
                            : std::vector<int64_t>{};
  auto recv_offset_list = primitive_->HasAttr(kAttrRecvOffsetList)
                            ? GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrRecvOffsetList))
                            : std::vector<int64_t>{};
  if (!send_offset_list.empty() && send_offset_list.size() != send_numel_list.size()) {
    MS_LOG(ERROR) << "Size of " << kAttrSendOffsetList << " should be equal to size of " << kAttrSendNumelList
                  << " for AlltoAllV, but got " << kAttrSendOffsetList << " size[" << send_offset_list.size()
                  << "] and " << kAttrSendNumelList << " size[" << send_numel_list.size() << "].";
    return false;
  }
  if (!recv_offset_list.empty() && recv_offset_list.size() != recv_numel_list.size()) {
    MS_LOG(ERROR) << "Size of " << kAttrRecvOffsetList << " should be equal to size of " << kAttrRecvNumelList
                  << " for AlltoAllV, but got " << kAttrRecvOffsetList << " size[" << recv_offset_list.size()
                  << "] and " << kAttrRecvNumelList << " size[" << recv_numel_list.size() << "].";
    return false;
  }
  params_.sendcounts.clear();
  params_.sdispls.clear();
  params_.recvcounts.clear();
  params_.rdispls.clear();
  uint64_t offset = 0;
  for (size_t i = 0; i < send_numel_list.size(); i++) {
    auto count = LongToSize(send_numel_list[i]);
    params_.sendcounts.push_back(count);
    if (send_offset_list.empty()) {
      params_.sdispls.push_back(offset);
      offset += count;
    } else {
      params_.sdispls.push_back(LongToSize(send_offset_list[i]));
    }
  }
  offset = 0;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    auto count = LongToSize(recv_numel_list[i]);
    params_.recvcounts.push_back(count);
    if (recv_offset_list.empty()) {
      params_.rdispls.push_back(offset);
      offset += count;
    } else {
      params_.rdispls.push_back(LongToSize(recv_offset_list[i]));
    }
  }
  return true;
}

bool HcomAlltoAllVKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  std::vector<KernelTensor *> temp;
  temp.push_back(inputs[0]);
  if (!HcclKernel::Init(temp, outputs)) {
    MS_LOG(ERROR) << "HcclKernel Init failed.";
    return false;
  }

  if (hccl_data_type_list_.empty()) {
    auto recv_type = GetValue<TypePtr>(primitive_->GetAttr(kAttrRecvType));
    if (recv_type == nullptr) {
      MS_LOG(ERROR) << "AlltoAllV got empty data type list and recv_type attr.";
      return false;
    }
    data_type_ = HcomUtil::ConvertHcclType(recv_type->type_id());
  } else {
    data_type_ = hccl_data_type_list_[0];
  }
  return true;
}

int HcomAlltoAllVKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  std::vector<int64_t> send_numel_list;
  std::vector<int64_t> recv_numel_list;
  if (inputs.size() == kInputNum3) {
    send_numel_list = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
    recv_numel_list = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl AlltoAllV input size " << inputs.size();
  }
  auto block_size = GetValue<int64_t>(primitive_->GetAttr(kAttrBlockSize));
  for (size_t i = 0; i < send_numel_list.size(); i++) {
    send_numel_list[i] = send_numel_list[i] * block_size;
    recv_numel_list[i] = recv_numel_list[i] * block_size;
  }
  if (!GetAllToAllVParam(send_numel_list, recv_numel_list)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetAllToAllVParam failed.";
  }

  int64_t output_numel = 0;
  ShapeVector shape;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    output_numel += recv_numel_list[i];
  }
  if (output_numel != 0) {
    shape.push_back(output_numel);
  }

  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), shape, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  if (!outputs.empty()) {
    output_size_list_.push_back(size);
  }
  return KRET_OK;
}

bool HcomAlltoAllVKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto send_tensor = inputs[0];
  MS_EXCEPTION_IF_NULL(send_tensor);
  auto send_buf = send_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(send_buf);
  void *recv_buf = nullptr;
  if (!outputs.empty()) {  // may be empty output when AlltoAllV is from NeighborExchangeV2
    auto recv_tensor = outputs[0];
    MS_EXCEPTION_IF_NULL(recv_tensor);
    recv_buf = recv_tensor->device_ptr();
    MS_EXCEPTION_IF_NULL(recv_buf);
  }
  if (NeedReGetHcom()) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need re-get hcom";
    comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  }
  auto hccl_result =
    hccl::HcclAdapter::GetInstance().HcclAlltoAllV(send_buf, recv_buf, params_, data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAlltoAllV failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
