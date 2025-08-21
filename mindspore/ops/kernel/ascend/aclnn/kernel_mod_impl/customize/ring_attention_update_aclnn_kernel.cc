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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/ring_attention_update_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "infer/ops_func_impl/ring_attention_update.h"

namespace mindspore {
namespace kernel {
void RingAttentionUpdateAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto layout = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  auto layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout);
  std::shared_ptr<KernelTensor> actual_seq_qlen_tensor;
  auto actual_seq_qlen_ptr = inputs[kIndex6];
  if (layout_str == "SBH") {
    auto shape = std::make_shared<abstract::TensorShape>(ShapeVector{});
    auto type = kInt64;
    auto value = MakeValue(0);
    actual_seq_qlen_tensor = std::make_shared<KernelTensor>(shape, type, value);
    actual_seq_qlen_ptr = actual_seq_qlen_tensor.get();
  } else if (layout_str == "TND") {
    if (inputs[kIndex6]->GetType()->type_id() == kMetaTypeNone) {
      MS_LOG(EXCEPTION) << "For 'RingAttentionUpdate', 'actual_seq_qlen' cannot be None when layout is TND.";
    }
  } else {
    MS_EXCEPTION(ValueError) << "For RingAttentionUpdate, the value of 'layout' must be SBH/TND.";
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        inputs[kIndex5], actual_seq_qlen_ptr, layout_str, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool RingAttentionUpdateAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto layout = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  auto layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout);
  std::shared_ptr<KernelTensor> actual_seq_qlen_tensor;
  auto actual_seq_qlen_ptr = inputs[kIndex6];
  if (layout_str == "SBH") {
    auto shape = std::make_shared<abstract::TensorShape>(ShapeVector{});
    auto type = kInt64;
    auto value = MakeValue(0);
    actual_seq_qlen_tensor = std::make_shared<KernelTensor>(shape, type, value);
    actual_seq_qlen_ptr = actual_seq_qlen_tensor.get();
  } else if (layout_str == "TND") {
    if (inputs[kIndex6]->GetType()->type_id() == kMetaTypeNone) {
      MS_LOG(EXCEPTION) << "For 'RingAttentionUpdate', 'actual_seq_qlen' cannot be None when layout is TND.";
    }
  } else {
    MS_EXCEPTION(ValueError) << "For RingAttentionUpdate, the value of 'layout' must be SBH/TND.";
  }
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        inputs[kIndex5], actual_seq_qlen_ptr, layout_str, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RingAttentionUpdate, RingAttentionUpdateAscend);
}  // namespace kernel
}  // namespace mindspore
