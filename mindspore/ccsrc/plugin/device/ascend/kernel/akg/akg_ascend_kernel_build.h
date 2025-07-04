/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_

#include <string>
#include "kernel/graph_kernel/akg/akg_kernel_build.h"
#include "kernel/graph_kernel/graph_kernel_builder_manager.h"

namespace mindspore {
namespace kernel {
class AkgAscendKernelBuilder : public AkgKernelBuilder {
 public:
  AkgAscendKernelBuilder() = default;
  ~AkgAscendKernelBuilder() = default;

  kernel::KernelBuildClient *GetClient() override { return &(kernel::AkgKernelBuildClient::Instance()); }
  void LoadCache() override { return; }
  KernelPackPtr SearchKernelCache(const std::string &kernel_name) override;
  KernelPackPtr InsertKernelCache(const std::string &kernel_name) override;
  void SetKernelMod(const KernelPackPtr &kernel_pack, const GraphKernelJsonGenerator &json_generator,
                    const AnfNodePtr &anf_node) override;
  void SaveJsonInfo(const string &kernel_name, const string &kernel_json) override;

 private:
  std::string GetPlatform() const override { return "ASCEND"; }
};

REG_GRAPH_KERNEL_BUILDER(kAscendDevice, false, AkgAscendKernelBuilder);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
