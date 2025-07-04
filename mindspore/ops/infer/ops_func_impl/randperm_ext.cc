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

#include <memory>
#include <limits>
#include "abstract/dshape.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ops/op_def/op_name.h"
#include "infer/ops_func_impl/randperm_ext.h"

namespace mindspore::ops {
namespace {
bool CheckForRandpermExtOverflow(TypeId tpyeId, int64_t n) {
  constexpr int64_t max_float16 = 65504;
  int64_t max = 0;

  switch (tpyeId) {
    case kNumberTypeUInt8:
      max = static_cast<int64_t>(std::numeric_limits<uint8_t>::max()) + 1;
      break;

    case kNumberTypeInt8:
      max = static_cast<int64_t>(std::numeric_limits<int8_t>::max()) + 1;
      break;

    case kNumberTypeInt16:
      max = static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1;
      break;

    case kNumberTypeInt32:
      max = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
      break;

    case kNumberTypeInt64:
      max = std::numeric_limits<int64_t>::max();
      break;

    case kNumberTypeFloat16:
      max = max_float16 + 1;
      break;

    // float32 and float64 do not overflow
    default:
      max = std::numeric_limits<int64_t>::max();
      break;
  }
  return n > max;
}
}  // namespace

ShapeArray RandpermExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<InferInfoPtr> &input_infos) const {
  const auto &prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckTypeIdValid("n", input_infos[kInputIndex0]->GetType(), {kNumberTypeInt64},
                                               prim_name);

  auto n_opt = input_infos[kInputIndex0]->GetScalarValue<int64_t>();
  auto type_opt = input_infos[kInputIndex3]->GetScalarValue<int64_t>();
  if (!n_opt.has_value() || !type_opt.has_value()) {
    return {{abstract::Shape::kShapeDimAny}};
  }

  int64_t n = n_opt.value();
  if (n <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input 'n' must be >= 0, but got data: " << n << ".";
  }
  auto out_type = static_cast<TypeId>(type_opt.value());
  if (CheckForRandpermExtOverflow(out_type, n)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the value of 'n' [" << n << "] exceeds limit of 'dtype' ["
                             << TypeIdToString(out_type) << "]. ";
  }
  return {{n}};
}

std::vector<TypeId> RandpermExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const std::vector<InferInfoPtr> &input_infos) const {
  auto dtype_opt = input_infos[kInputIndex3]->GetScalarValue<int64_t>();
  TypeId output_type{kNumberTypeInt64};
  if (dtype_opt.has_value()) {
    output_type = static_cast<TypeId>(dtype_opt.value());
  }
  const std::set<TypeId> output_valid_types = {kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeInt16,
                                               kNumberTypeInt8,    kNumberTypeUInt8,   kNumberTypeFloat16,
                                               kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};
  (void)CheckAndConvertUtils::CheckTypeIdValid("dtype", output_type, output_valid_types, primitive->name());
  return {output_type};
}
}  // namespace mindspore::ops
