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

#include "kernel/gpu/pyboost/customize/pow_scalar_tensor.h"
#include "kernel/gpu/pyboost/auto_generate/pow.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void PowScalarTensorGPUCustomize(const std::shared_ptr<OpRunner> &op, const ScalarPtr &input,
                                 const TensorPtr &exponent_tensor) {
  MS_LOG(DEBUG) << "PowScalarTensor Call start";
  OpRunner::InferOpOutput(op, input, exponent_tensor);

  // the PowScalarTensor primitive does not support GPU, so use Pow instead.
  const auto pow_op = CREATE_PYBOOST_OP(Pow, device::DeviceType::kGPU);

  // handle type promotion manually since the GPU kernelmod Pow does not support it
  const auto out_dtype = op->output(0)->Dtype();
  auto exp_tensor_cast = exponent_tensor;
  if (exponent_tensor->Dtype()->type_id() != out_dtype->type_id()) {
    exp_tensor_cast = PyBoostUtils::CastTensor(exponent_tensor, out_dtype->type_id(), device::DeviceType::kGPU);
  }
  const auto input_tensor = PyBoostUtils::ScalarToTensor(input, out_dtype);

  const auto pow_out = pow_op->Call(input_tensor, exp_tensor_cast);
  op->set_outputs({pow_out});

  MS_LOG(DEBUG) << "PowScalarTensor Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
