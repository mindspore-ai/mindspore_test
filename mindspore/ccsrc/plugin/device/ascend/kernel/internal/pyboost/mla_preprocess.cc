/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/pyboost/mla_preprocess.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {

internal::InternalOpPtr MlaPreprocess::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                    const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateMlaPreprocessOp(inputs, outputs, param_, internal::kInternalMlaPreprocessOpName);
}

void MlaPreprocess::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                         const uint64_t &tiling_key, const TensorPtr &input1, const TensorPtr &gamma1,
                         const TensorPtr &beta1, const TensorPtr &quant_scale1, const TensorPtr &quant_offset1,
                         const TensorPtr &wdqkv, const TensorPtr &bias1, const TensorPtr &gamma2,
                         const TensorPtr &beta2, const TensorPtr &quant_scale2, const TensorPtr &quant_offset2,
                         const TensorPtr &gamma3, const TensorPtr &sin1, const TensorPtr &cos1, const TensorPtr &sin2,
                         const TensorPtr &cos2, const TensorPtr &key_cache, const TensorPtr &slot_mapping,
                         const TensorPtr &wuq, const TensorPtr &bias2, const TensorPtr &slot_wuk,
                         const TensorPtr &de_scale1, const TensorPtr &de_scale2, const TensorPtr &ctkv_scale,
                         const TensorPtr &qnope_scale, const TensorPtr &krope_cache, const int64_t &param_cache_mode) {
  TensorPtrList inputs = {input1,    gamma1,    beta1,        quant_scale1,  quant_offset1, wdqkv, bias1,
                          gamma2,    beta2,     quant_scale2, quant_offset2, gamma3,        sin1,  cos1,
                          sin2,      cos2,      key_cache,    slot_mapping,  wuq,           bias2, slot_wuk,
                          de_scale1, de_scale2, ctkv_scale,   qnope_scale,   krope_cache};
  TensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);
  param_.n = input1->shape_c()[0];
  param_.head_num = slot_wuk->shape_c()[0];
  param_.cache_mode = param_cache_mode;

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(MlaPreprocess, MlaPreprocess);
}  // namespace kernel
}  // namespace mindspore
