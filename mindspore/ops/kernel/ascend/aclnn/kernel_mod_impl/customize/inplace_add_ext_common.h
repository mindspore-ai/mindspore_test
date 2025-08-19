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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INPLACE_ADD_EXT_COMMON_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INPLACE_ADD_EXT_COMMON_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace inplace_add_ext {
static void GetScalarFromInput(KernelTensor *inputPtr, ScalarPtr *dstScalar) {
  auto scalar_dtype_id = inputPtr->dtype_id();
  switch (scalar_dtype_id) {
    case kNumberTypeBool: {
      auto scalar_value = inputPtr->GetValueWithCheck<bool>();
      MAKE_SCALAR(scalar_value, scalar_dtype_id, (*dstScalar));
      break;
    }
    case kNumberTypeFloat32: {
      double scalar_value = static_cast<double>((inputPtr->GetValueWithCheck<float>()));
      MAKE_SCALAR(scalar_value, kNumberTypeFloat64, (*dstScalar));
      break;
    }
    case kNumberTypeFloat64: {
      auto scalar_value = inputPtr->GetValueWithCheck<double>();
      MAKE_SCALAR(scalar_value, scalar_dtype_id, (*dstScalar));
      break;
    }
    case kNumberTypeInt32: {
      auto scalar_value = inputPtr->GetValueWithCheck<int32_t>();
      MAKE_SCALAR(scalar_value, scalar_dtype_id, (*dstScalar));
      break;
    }
    case kNumberTypeInt64: {
      auto scalar_value = inputPtr->GetValueWithCheck<int64_t>();
      MAKE_SCALAR(scalar_value, scalar_dtype_id, (*dstScalar));
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "InplaceAdd only support bool, float32, float64, int32 and int64, but got:"
                        << scalar_dtype_id;
  }
}

class InplaceAddExtAclnnKernelMod : public AclnnKernelMod {
 public:
  InplaceAddExtAclnnKernelMod() : AclnnKernelMod(std::move("aclnnInplaceAdd")) {}
  ~InplaceAddExtAclnnKernelMod() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  ScalarPtr alpha_scalar_ = nullptr;
};

class InplaceAddsExtAclnnKernelMod : public AclnnKernelMod {
 public:
  InplaceAddsExtAclnnKernelMod() : AclnnKernelMod(std::move("aclnnInplaceAdds")) {}
  ~InplaceAddsExtAclnnKernelMod() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  ScalarPtr alpha_scalar_ = nullptr;
  ScalarPtr other_scalar_ = nullptr;
};
}  // namespace inplace_add_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INPLACE_ADD_EXT_COMMON_ACLNN_KERNEL_MOD_H_
