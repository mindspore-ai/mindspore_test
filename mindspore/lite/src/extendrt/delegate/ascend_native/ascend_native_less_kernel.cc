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

#include "extendrt/delegate/ascend_native/ascend_native_less_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"

namespace mindspore::kernel {
using mindspore::ops::kNameLess;

int AscendNativeLessKernel::InferShape() {
  if (out_tensors_[0]->shape().size() == 0) {
    if (in_tensors_[0] != nullptr) out_tensors_[0]->set_shape(in_tensors_[0]->shape());
  }
  return kSuccess;
}

int AscendNativeLessKernel::Prepare() { return kSuccess; }

int AscendNativeLessKernel::Run() {
  MS_LOG(INFO) << "AscendNativeLessKernel::Execute";

  return kSuccess;
}

int AscendNativeLessKernel::ReSize() { return kSuccess; }

REGISTER_ASCEND_NATIVE_CREATOR(kNameLess, AscendNativeLessKernel)
}  // namespace mindspore::kernel
