/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.h"
#include <memory>
#include "kernel/framework_utils.h"
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void AkgCpuKernelBuilder::SetKernelMod(const KernelPackPtr &kernel_pack, const GraphKernelJsonGenerator &json_generator,
                                       const AnfNodePtr &anf_node) {
  auto kernel_mod_ptr = std::make_shared<AkgCpuKernelMod>(kernel_pack);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}

void AkgCpuKernelBuilder::SaveJsonInfo(const string &kernel_name, const string &kernel_json) {
  kernel::SaveJsonInfo(kernel_name, kernel_json, kernel::KernelMeta::GetInstance()->kernel_meta_path());
}
}  // namespace kernel
}  // namespace mindspore
