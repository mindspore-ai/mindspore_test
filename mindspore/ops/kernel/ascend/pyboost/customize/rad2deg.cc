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

#include <string>
#include "kernel/ascend/pyboost/customize/rad2deg.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void Rad2degAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Rad2deg Launch start";
  constexpr double M_180_PI = 57.295779513082320876798154814105170332405472466564;
  static const std::vector<TypeId> supported_dtypes = {
    kNumberTypeBool,  kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
    kNumberTypeUInt8, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};
  ScalarPtr other = nullptr;
  TypeId input_type = input_tensor->data_type();
  bool is_supported = std::any_of(supported_dtypes.begin(), supported_dtypes.end(),
                                  [&input_type](const TypeId &type) { return input_type == type; });

  if (!is_supported) {
    MS_EXCEPTION(TypeError) << "For `Rad2deg`, the dtype of `input` is not supported.";
  }
  other = std::make_shared<FP64Imm>(M_180_PI);
  auto muls_out = muls(input_tensor, other);
  op->set_outputs({muls_out});
  MS_LOG(DEBUG) << "Rad2deg Launch end";

  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
