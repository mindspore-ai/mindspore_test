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

#include "kernel/ascend/pyboost/customize/softsign_ext.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void SoftsignExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "SoftsignExt Launch start";
  auto type_id = input_tensor->data_type();
  const std::unordered_set<TypeId> valid_types = {
    kNumberTypeBool,  kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64, kNumberTypeUInt8,
    kNumberTypeFloat, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeDouble};
  if (valid_types.find(type_id) == valid_types.end()) {
    MS_LOG(EXCEPTION) << "For 'Softsign', the type of 'input' must be Tensor[Bool, Int8, Int16, Int32, Int64, UInt8, "
                         "Float16, Float32, Float64], but got "
                      << input_tensor->Dtype();
  }
  auto output_tensor = abs(input_tensor);
  output_tensor = add_scalar(output_tensor, std::make_shared<Int64Imm>(1), std::make_shared<Int64Imm>(1));
  output_tensor = div(input_tensor, output_tensor);
  op->set_outputs({output_tensor});
  MS_LOG(DEBUG) << "SoftsignExt Launch end";
  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
