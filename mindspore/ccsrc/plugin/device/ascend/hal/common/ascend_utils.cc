/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include "utils/dlopen_macro.h"
#include "acl/error_codes/rt_error_codes.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/common/debug/common.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"

namespace mindspore {
namespace device {
namespace ascend {
// There are 2 requirements for enabling collective communication in DVM
// 1. Jit level is set to O1
// 2. At least one of the collective communication primitives is included in `enable_cluster_ops` or
// `enable_cluster_ops_only` of graph kernel flags
bool EnableDvmComm() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b") {
    return false;
  }

  const auto &jit_level = MsContext::GetInstance()->GetJitLevel();
  if (jit_level != "O1") {
    return false;
  }
  const auto &gk_flags = graphkernel::GraphKernelFlags::GetInstance();
  const auto &enable_cluster_ops = gk_flags.enable_cluster_ops;
  const auto &enable_cluster_ops_only = gk_flags.enable_cluster_ops_only;
  auto check_func = [](const std::string &op) {
    return op == "AllReduce" || op == "AllGather" || op == "ReduceScatter";
  };
  if (std::any_of(enable_cluster_ops.begin(), enable_cluster_ops.end(), check_func)) {
    return true;
  }
  if (std::any_of(enable_cluster_ops_only.begin(), enable_cluster_ops_only.end(), check_func)) {
    return true;
  }
  return false;
}

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
