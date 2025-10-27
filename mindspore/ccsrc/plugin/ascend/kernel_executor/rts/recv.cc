/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/rts/recv.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {

RecvKernel::~RecvKernel() {}

bool RecvKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!common::AnfAlgo::HasNodeAttr(kAttrEventId, anf_node->cast<CNodePtr>())) {
    MS_LOG(INTERNAL_EXCEPTION) << "RecvKernel has no attr kAttrEventId";
  }
  event_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));
  record_stream_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrRecordEventStream));

  if (common::AnfAlgo::HasNodeAttr(kAttrWaitEvent, anf_node->cast<CNodePtr>())) {
    event_ = reinterpret_cast<aclrtEvent>(GetValue<uintptr_t>(primitive->GetAttr(kAttrWaitEvent)));
  }
  MS_LOG(INFO) << "recv op event_id_: " << event_id_ << ", record_stream_id_ : " << record_stream_id_ << ".";
  return true;
}

bool RecvKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                        const std::vector<KernelTensor *> &, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto status = CALL_ASCEND_API(aclrtStreamWaitEvent, stream_ptr, event_);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Recv aclrtStreamWaitEvent failed!";
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
