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

#include "infer/ops_func_impl/logaddexp2.h"

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>
#include "op_def/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
ShapeArray LogAddExp2FuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto output_shape = input_infos[kInputIndex0]->GetShape();
  std::vector<std::string> input_names = {"input", "other"};
  for (size_t i = 1; i < input_infos.size(); ++i) {
    auto input_shape = input_infos[i]->GetShape();
    output_shape = CalBroadCastShape(output_shape, input_shape, primitive->name(), input_names[i - 1], input_names[i]);
  }
  return {output_shape};
}

std::vector<TypeId> LogAddExp2FuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex0]);
  return {input_infos[kInputIndex0]->GetType()};
}

}  // namespace mindspore::ops
