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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nan_to_num_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace nan_to_num {
void NanToNumAscend::GetInfValues(TypeId input_type, const std::optional<float> &posinf,
                                  const std::optional<float> &neginf, bool posinf_has_value, bool neginf_has_value) {
  const float DOUBLE_MAX_VALUE = 1.7976931348623157e+308;
  const float DOUBLE_MIN_VALUE = -1.7976931348623157e+308;
  const float FLOAT32_MAX_VALUE = 3.4028235e+38;
  const float FLOAT32_MIN_VALUE = -3.4028235e+38;
  const float FLOAT16_MAX_VALUE = 65504.0;
  const float FLOAT16_MIN_VALUE = -65504.0;
  const float BFLOAT16_MAX_VALUE = 3.3895314e+38;
  const float BFLOAT16_MIN_VALUE = -3.3895314e+38;
  switch (input_type) {
    case kNumberTypeFloat64:
      posinf_ = posinf_has_value ? posinf.value() : DOUBLE_MAX_VALUE;
      neginf_ = neginf_has_value ? neginf.value() : DOUBLE_MIN_VALUE;
      break;
    case kNumberTypeFloat32:
      posinf_ = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
      neginf_ = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
      break;
    case kNumberTypeFloat16:
      posinf_ = posinf_has_value ? posinf.value() : FLOAT16_MAX_VALUE;
      neginf_ = neginf_has_value ? neginf.value() : FLOAT16_MIN_VALUE;
      break;
    case kNumberTypeBFloat16:
      posinf_ = posinf_has_value ? posinf.value() : BFLOAT16_MAX_VALUE;
      neginf_ = neginf_has_value ? neginf.value() : BFLOAT16_MIN_VALUE;
      break;
    default:
      posinf_ = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
      neginf_ = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
      break;
  }
}

void NanToNumAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  const float DEFAULT_NAN = 0.0;

  auto nan = inputs[kIndex1]->GetOptionalValueWithCheck<float>();
  nan_ = nan.has_value() ? nan.value() : DEFAULT_NAN;

  auto posinf = inputs[kIndex2]->GetOptionalValueWithCheck<float>();
  auto neginf = inputs[kIndex3]->GetOptionalValueWithCheck<float>();

  bool posinf_has_value = posinf.has_value();
  bool neginf_has_value = neginf.has_value();
  if (posinf_has_value && neginf_has_value) {
    posinf_ = device::ascend::ConvertKernelTensor<float>(inputs[kIndex2]);
    neginf_ = device::ascend::ConvertKernelTensor<float>(inputs[kIndex3]);
  } else {
    auto input_type = inputs[kIndex0]->dtype_id();
    GetInfValues(input_type, posinf, neginf, posinf_has_value, neginf_has_value);
  }
  GetWorkspaceForResize(inputs[kIndex0], nan_, posinf_, neginf_, outputs[kIndex0]);
}

bool NanToNumAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], nan_, posinf_, neginf_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NanToNum, NanToNumAscend);
}  // namespace nan_to_num
}  // namespace kernel
}  // namespace mindspore
