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

#include "plugin/ascend/kernel_executor/rts/res_limit.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
ResLimitKernel::~ResLimitKernel() {}

bool ResLimitKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (common::AnfAlgo::HasNodeAttr(kAttrCubeNum, anf_node->cast<CNodePtr>())) {
    res_limit_map_[aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE] =
      GetValue<uint32_t>(primitive->GetAttr(kAttrCubeNum));
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrVectorNum, anf_node->cast<CNodePtr>())) {
    res_limit_map_[aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE] =
      GetValue<uint32_t>(primitive->GetAttr(kAttrVectorNum));
  }
  return true;
}

bool ResLimitKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                            const std::vector<KernelTensor *> &, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  for (const auto &iter : res_limit_map_) {
    auto ret = CALL_ASCEND_API(aclrtSetStreamResLimit, stream_ptr, iter.first, iter.second);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtSetStreamResLimit failed! Error flag is " << ret;
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
