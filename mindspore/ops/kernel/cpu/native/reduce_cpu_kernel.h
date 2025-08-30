/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <complex>
#include "base/bfloat16.h"
#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/utils/cpu_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

#define REDUCE_DO_BOOL_CAST(TYPE_ID, TYPE)                   \
  case TYPE_ID:                                              \
    CastKernelTensor<TYPE, bool>(input_tensor, cast_tensor); \
    break

namespace mindspore {
namespace kernel {
namespace reduce_cpu {
class ReduceCpuKernelMod : public NativeCpuKernelMod {
 public:
  ReduceCpuKernelMod() = default;
  explicit ReduceCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ReduceCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    constexpr auto kReduceAll = "ReduceAll";
    if (kernel_type_ == kReduceAll) {
      return CastBoolLaunch(inputs, workspace, outputs);
    }
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  bool CastBoolLaunch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs) {
    auto input_type = inputs[kIndex0]->dtype_id();
    auto input_tensor = inputs[kIndex0];
    auto cast_tensor = workspace[kIndex0];
    if (CheckShapeNull(input_tensor->GetDeviceShapeVector(), kernel_type_, "input")) {
      return func_obj_->RunFunc(inputs, workspace, outputs);
    }
    switch (input_type) {
      REDUCE_DO_BOOL_CAST(kNumberTypeUInt8, uint8_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeUInt16, uint16_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeUInt32, uint32_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeUInt64, uint64_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeInt8, int8_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeInt16, int16_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeInt32, int32_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeInt64, int64_t);
      REDUCE_DO_BOOL_CAST(kNumberTypeFloat16, Eigen::half);
      REDUCE_DO_BOOL_CAST(kNumberTypeFloat32, float);
      REDUCE_DO_BOOL_CAST(kNumberTypeFloat64, double);
      REDUCE_DO_BOOL_CAST(kNumberTypeBFloat16, bfloat16);
      REDUCE_DO_BOOL_CAST(kNumberTypeComplex64, std::complex<float>);
      REDUCE_DO_BOOL_CAST(kNumberTypeComplex128, std::complex<double>);
      case kNumberTypeBool:
        cast_tensor = input_tensor;
        break;
      default:
        break;
    }
    auto casted_inputs = inputs;
    casted_inputs[kIndex0] = cast_tensor;
    return func_obj_->RunFunc(casted_inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{"Unknown"};
};
}  // namespace reduce_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_
