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

#include "kernel/ascend/pyboost/customize/count_nonzero.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/ascend/pyboost/auto_generate/sum_ext.h"
#include "kernel/ascend/pyboost/auto_generate/ne_scalar.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr CountNonZeroAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const BaseTensorPtr &input_tensor,
                                                  const std::optional<ValueTuplePtr> &dims) {
  ScalarPtr other;
  TypeId other_type = input_tensor->data_type();
  if (other_type == kNumberTypeComplex || other_type == kNumberTypeComplex64 || other_type == kNumberTypeComplex128) {
    MAKE_SCALAR(0, kNumberTypeInt8, other);
  } else {
    MAKE_SCALAR(0, other_type, other);
  }
  BoolImmPtr keep_dims = std::make_shared<BoolImm>(false);
  Int64ImmPtr out_dtype = std::make_shared<Int64Imm>(kNumberTypeInt64);
  auto device_context = op->device_context();

  const auto &device_name = device_context->device_context_key_.device_name_;

  auto nescalar_op = CREATE_PYBOOST_OP(NeScalar, device_name);
  auto ne_tensor = nescalar_op->Call(input_tensor, other);

  auto reducesum_op = CREATE_PYBOOST_OP(SumExt, device_name);
  auto output_tensor = reducesum_op->Call(ne_tensor, dims, keep_dims, out_dtype);
  op->set_outputs({output_tensor});
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
