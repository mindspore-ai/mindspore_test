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
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/opapi/aclnn/index_fill_tensor_aclnn_kernel.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {

void IndexFillTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  dim_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  value_ = transform::ConvertKernelTensor<ScalarPtr>(inputs[kIndex3]);
  std::vector<int64_t> convertedIndex;
  auto alpha_dtype_id = inputs[kIndex2]->dtype_id();
  switch (alpha_dtype_id) {
    case kNumberTypeInt64: {
      auto index = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt32: {
      auto index = transform::ConvertKernelTensor<std::vector<int32_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt16: {
      auto index = transform::ConvertKernelTensor<std::vector<int16_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt8: {
      auto index = transform::ConvertKernelTensor<std::vector<int8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeUInt8: {
      auto index = transform::ConvertKernelTensor<std::vector<uint8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "IndexAddExt only support int32, int64, int16, int8, and uint8, but got "
                        << TypeIdToString(alpha_dtype_id);
  }
  GetWorkspaceForResize(inputs[kIndex0], dim_, convertedIndex, value_, outputs[kIndex0]);
}

bool IndexFillTensorAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto alpha_dtype_id = inputs[kIndex2]->dtype_id();
  std::vector<int64_t> convertedIndex;
  switch (alpha_dtype_id) {
    case kNumberTypeInt64: {
      auto index = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt32: {
      auto index = transform::ConvertKernelTensor<std::vector<int32_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt16: {
      auto index = transform::ConvertKernelTensor<std::vector<int16_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt8: {
      auto index = transform::ConvertKernelTensor<std::vector<int8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeUInt8: {
      auto index = transform::ConvertKernelTensor<std::vector<uint8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "IndexAddExt only support int32, int64, int16, int8, and uint8, but got "
                        << TypeIdToString(alpha_dtype_id);
  }
  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, convertedIndex, value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(IndexFillTensor, IndexFillTensorAscend);
}  // namespace kernel
}  // namespace mindspore
