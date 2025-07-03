/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/acl/acl_kernel_build.h"
#include <vector>
#include <string>
#include "kernel/ascend/acl/acl_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/framework_utils.h"
#include "utils/trace_base.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
KernelModPtr AclOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  MS_LOG(INFO) << "Begin to create acl kernel module for primitive " << primitive->name();

  auto kernel_mod_ptr = std::make_shared<AclKernelMod>();
  if (common::AnfAlgo::IsGetNextNode(anf_node)) {
    kernel_mod_ptr = Factory<AclKernelMod>::Instance().Create("GetNext");
  } else if (primitive->name() == "Custom") {
    kernel_mod_ptr = Factory<AclKernelMod>::Instance().Create("CustomOp");
  }
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_mod_ptr)
         ->Init(primitive, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize acl kernel op[" << anf_node->fullname_with_scope() << "] failed."
      << trace::DumpSourceLines(anf_node);
  }

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
  MS_EXCEPTION_IF_NULL(build_info);
  auto input_formats = build_info->GetAllInputFormats();
  auto input_types = build_info->GetAllInputDeviceTypes();
  auto output_formats = build_info->GetAllOutputFormats();
  auto output_types = build_info->GetAllOutputDeviceTypes();
  kernel_mod_ptr->SetDeviceInfo(input_formats, output_formats, input_types, output_types);

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // acl_kernel_mod use proto value_depend indices
  kernel_mod_ptr->SetValueDependArgs(abstract::GetValueDependArgIndices(cnode, true));
  if (common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
    kernel_mod_ptr->SetDynamic(true);
    return kernel_mod_ptr;
  }

  if (kernel::CheckResizeCondition(cnode)) {
    kernel_mod_ptr->SetDynamic(false);
    kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors);
  }

  MS_LOG(INFO) << "Finished creating acl kernel module for primitive " << primitive->name();
  return kernel_mod_ptr;
}

KernelModPtr CreateAclKernelMod(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) {
  MS_LOG(INFO) << "Begin to create acl kernel module for primitive " << primitive->name();
  auto kernel_mod_ptr = std::make_shared<AclKernelMod>();
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

  std::vector<std::string> input_formats;
  std::vector<std::string> output_formats;
  std::vector<mindspore::TypeId> input_dtypes;
  std::vector<mindspore::TypeId> output_dtypes;
  for (auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    input_formats.emplace_back(input->GetStringFormat());
    input_dtypes.emplace_back(input->dtype_id());
  }
  for (auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    output_formats.emplace_back(output->GetStringFormat());
    output_dtypes.emplace_back(output->dtype_id());
  }

  kernel_mod_ptr->SetDeviceInfo(input_formats, output_formats, input_dtypes, output_dtypes);
  kernel_mod_ptr->SetDynamic(false);

  // acl_kernel_mod use proto value_depend indices
  kernel_mod_ptr->SetValueDependArgs(primitive->name(),
                                     abstract::GetValueDependArgIndicesFromProto(primitive, inputs.size()));

  MS_LOG(INFO) << "Finished creating acl kernel module for primitive " << primitive->name();
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
