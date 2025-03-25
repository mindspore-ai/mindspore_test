/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_CACHE_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_CACHE_H_
#include <vector>
#include <string>
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "kernel/ascend/acl_ir/custom/custom_aclnn_utils.h"

namespace mindspore::device::ascend {
using CustomSupportType = mindspore::kernel::custom::CustomSupportType;
bool CustomHitCacheSingle(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, uint64_t *hash_id,
                          const std::vector<std::vector<KernelTensor *>> &inputs,
                          const std::vector<std::vector<KernelTensor *>> &outputs,
                          const std::vector<CustomSupportType> &input_output_types);
uint64_t CustomAclnnHash(const std::string &op_type, const std::vector<std::vector<KernelTensor *>> &inputs,
                         const std::vector<std::vector<KernelTensor *>> &outputs,
                         const std::vector<CustomSupportType> &input_output_types);
void CustomRefreshAddr(const std::string &op_type, const std::vector<std::vector<KernelTensor *>> &inputs,
                       const std::vector<std::vector<KernelTensor *>> &outputs,
                       const std::vector<CustomSupportType> &input_output_types);

}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_CACHE_H_
