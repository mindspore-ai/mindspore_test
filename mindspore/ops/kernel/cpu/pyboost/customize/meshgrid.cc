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

#include "mindspore/ops/kernel/cpu/pyboost/customize/meshgrid.h"

#include "ir/scalar.h"
#include "kernel/cpu/cpu_kernel.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/device_address_utils.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/customize/op_common.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "runtime/pipeline/pipeline.h"
#include "mindspore/ccsrc/pyboost/customize/meshgrid.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> MeshgridCPUCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const ValueTuplePtr &tensors_list,
                                                        const Int64ImmPtr &indexing) {
  MS_LOG(DEBUG) << "Nonzero CPU start";
  std::vector<tensor::BaseTensorPtr> output = MeshgridCustomizeCall(op, tensors_list, indexing, kCPUDevice);
  MS_LOG(DEBUG) << "NonZero CPU end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
