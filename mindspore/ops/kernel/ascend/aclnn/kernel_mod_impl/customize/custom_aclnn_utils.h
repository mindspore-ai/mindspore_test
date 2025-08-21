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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_CUSTOM_ACLNN_KERNEL_FACTORY_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_CUSTOM_ACLNN_KERNEL_FACTORY_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include "ops/base_operator.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_aclnn_kernel.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_v2_aclnn_kernel.h"
#include "mindspore/core/include/ir/tensor.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/custom_kernel.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
using CustomPyboostKernelMod = pyboost::CustomAclnnPyboostKernelModBase;
OPS_ASCEND_API std::string AddPrefixForCustomNode(const std::string &op_type, bool unset = false);
std::shared_ptr<AclnnKernelMod> GetCustomAclnnKernelMod(const AnfNodePtr &anf_node);
std::shared_ptr<AclnnKernelMod> GetCustomAclnnKernelMod(const std::string &op_type, size_t arg_num);
std::shared_ptr<CustomPyboostKernelMod> GetCustomAclnnPyboostKernelMod(const std::string &op_type, size_t arg_num);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_CUSTOM_ACLNN_KERNEL_FACTORY_H_
