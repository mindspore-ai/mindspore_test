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

#include "kernel/ascend/pyboost/customize/inplace_exponential.h"

#include <memory>
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

double GetScalarValue(const std::shared_ptr<Scalar> &scalar) {
  if (scalar->isa<Int32Imm>()) {
    return GetValue<int32_t>(scalar);
  } else if (scalar->isa<Int64Imm>()) {
    return GetValue<int64_t>(scalar);
  } else if (scalar->isa<FP32Imm>()) {
    return GetValue<float>(scalar);
  } else if (scalar->isa<FP64Imm>()) {
    return GetValue<double>(scalar);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported type: " << scalar->type_name();
  }
}

tensor::BaseTensorPtr InplaceExponentialAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                                        const ScalarPtr lambda, const BaseTensorPtr &seed,
                                                        const BaseTensorPtr &offset) {
  MS_LOG(DEBUG) << "InplaceExponentialAscendCustomize Call start";
  auto out = inplace_uniform(input, std::make_shared<FP64Imm>(0.0), std::make_shared<FP64Imm>(1.0), seed, offset);
  out = inplace_sub_scalar(out, std::make_shared<FP64Imm>(1.0), std::make_shared<FP64Imm>(1.0));
  out = inplace_muls(out, std::make_shared<FP64Imm>(-1.0));
  out = inplace_log(out);
  double lambda_val = GetScalarValue(lambda);
  out = inplace_divs(out, std::make_shared<FP64Imm>(-lambda_val));
  MS_LOG(DEBUG) << "InplaceExponentialAscendCustomize Call end";
  op->set_outputs({out});
  return out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
