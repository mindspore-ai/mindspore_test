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

#include "mindspore/ops/kernel/cpu/pyboost/customize/pow_tensor_scalar.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/pow.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void PowTensorScalarCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                 const ScalarPtr &exponent) {
  MS_LOG(DEBUG) << "PowTensorScalar Call start";
  OpRunner::InferOpOutput(op, input_tensor, exponent);

  // the PowTensorScalar primitive does not support CPU, so use Pow instead.
  const auto pow_op = CREATE_PYBOOST_OP(Pow, device::DeviceType::kCPU);

  // handle type promotion manually since the CPU kernelmod Pow does not support it
  const auto out_dtype = op->output(0)->Dtype();
  auto input_tensor_cast = input_tensor;
  if (input_tensor->Dtype()->type_id() != out_dtype->type_id()) {
    input_tensor_cast = PyBoostUtils::CastTensor(input_tensor, out_dtype->type_id(), device::DeviceType::kCPU);
  }
  const auto exponent_tensor = PyBoostUtils::ScalarToTensor(exponent, out_dtype);

  const auto pow_out = pow_op->Call(input_tensor_cast, exponent_tensor);
  op->set_outputs({pow_out});

  MS_LOG(DEBUG) << "PowTensorScalar Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
