/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_BUILD_H_

#include <memory>
#include <string>
#include <vector>

#include "kernel/kernel.h"
#include "ir/tensor.h"
#include "kernel/common/pyboost/op_runner.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace kernel {
KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node);
bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node);
void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                               std::vector<std::string> *output_formats);
void AcmeAscendCall(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<tensor::BaseTensorPtr> &inputs,
                    const std::vector<void *> &params);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_BUILD_H_
