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
#include "mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_scatter_value_reduce_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace inplace_scatter_value_reduce {

int64_t InplaceScatterValueReduceAscend::GetReduce(const std::vector<KernelTensor *> &inputs) {
  auto reduce = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  // 0 means 'none' (replace) in aclnn, but should use scatter_ without reduce instead of using 'none'
  if ((reduce != Reduce::ADD) && (reduce != Reduce::MULTIPLY)) {
    MS_EXCEPTION(ValueError) << "For InplaceScatterValueReduce, reduce must be either 'add' or 'multiply', but got: '"
                             << mindspore::device::ascend::ScatterReduceMode::ConvertEnumToString(reduce) << "'.";
  }
  return reduce;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceScatterValueReduce, InplaceScatterValueReduceAscend);
}  // namespace inplace_scatter_value_reduce
}  // namespace kernel
}  // namespace mindspore
