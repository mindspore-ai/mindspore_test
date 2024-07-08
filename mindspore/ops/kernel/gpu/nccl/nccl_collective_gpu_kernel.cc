/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/nccl/nccl_collective_gpu_kernel.h"
#include <utility>

namespace mindspore {
namespace kernel {
bool NcclCollectiveGpuKernel::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto dtype = inputs[0]->dtype_id();
  nccl_data_type_ = nccl_dtype(inputs[0]->dtype_id());
  InferCommType(kernel_name_, primitive_, dtype);

  unit_size_ = abstract::TypeIdSize(dtype);

  SelectCollectiveHandle();
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

const std::vector<std::pair<KernelAttr, NcclCollectiveGpuKernel::KernelRunFunc>> &NcclCollectiveGpuKernel::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NcclCollectiveGpuKernel::KernelRunFunc>> func_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &NcclCollectiveGpuKernel::LaunchKernel<bool>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &NcclCollectiveGpuKernel::LaunchKernel<int8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &NcclCollectiveGpuKernel::LaunchKernel<int32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &NcclCollectiveGpuKernel::LaunchKernel<int64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &NcclCollectiveGpuKernel::LaunchKernel<uint8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &NcclCollectiveGpuKernel::LaunchKernel<uint32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &NcclCollectiveGpuKernel::LaunchKernel<uint64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &NcclCollectiveGpuKernel::LaunchKernel<half>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &NcclCollectiveGpuKernel::LaunchKernel<float>},
  };
  return func_list;
}

ncclRedOp_t NcclCollectiveGpuKernel::GetNcclReduceOpType(const std::string &op_type) {
  auto iter = kNcclReduceOpTypeMap.find(op_type);
  if (iter == kNcclReduceOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: sum, max, min, prod currently, "
                      << "but got " << op_type;
  }
  return iter->second;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AllReduce, NcclCollectiveGpuKernel);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AllGather, NcclCollectiveGpuKernel);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ReduceScatter, NcclCollectiveGpuKernel);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Broadcast, NcclCollectiveGpuKernel);
}  // namespace kernel
}  // namespace mindspore
