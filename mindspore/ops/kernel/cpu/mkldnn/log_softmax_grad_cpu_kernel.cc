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

#include "kernel/cpu/mkldnn/log_softmax_grad_cpu_kernel.h"
#include <algorithm>
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogSoftmaxGradInputsNum = 3;
constexpr size_t kLogSoftmaxGradOutputsNum = 1;
}  // namespace

bool LogSoftmaxGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSoftmaxGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSoftmaxGradOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int LogSoftmaxGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  axis_ = inputs.at(kIndex2)->GetValueWithCheck<int64_t>();
  auto src_shape = inputs[0]->GetDeviceShapeVector();
  if (axis_ >= SizeToLong(src_shape.size())) {
    axis_ = SizeToLong(src_shape.size()) - 1;
  }
  while (axis_ < 0) {
    axis_ += SizeToLong(src_shape.size());
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  auto desc = CreateDesc<dnnl::logsoftmax_forward::desc>(dnnl::prop_kind::forward_training, src_desc, axis_);
  auto prim_desc = CreateDesc<dnnl::logsoftmax_forward::primitive_desc>(desc, engine_);
  // backward description
  auto backward_desc = CreateDesc<dnnl::logsoftmax_backward::desc>(src_desc, src_desc, axis_);
  auto backward_prim_desc = CreateDesc<dnnl::logsoftmax_backward::primitive_desc>(backward_desc, engine_, prim_desc);
  primitive_ = CreatePrimitive<dnnl::logsoftmax_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, src_desc);
  return ret;
}

bool LogSoftmaxGradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSoftmaxGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSoftmaxGradOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_DST, inputs[0]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[1]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->device_ptr());
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogSoftmaxGrad, LogSoftmaxGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
