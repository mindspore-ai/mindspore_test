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

#include "infer/ops_func_impl/swiglu_dynamic_quant.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <set>
#include <functional>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
abstract::BaseShapePtr SwiGLUDynamicQuantFuncImpl::InferShape(const PrimitivePtr &prim,
                                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();

  auto is_x_static = !IsDynamic(x_shape);
  constexpr size_t kSplitNum = 2;
  auto x_rank = SizeToLong(x_shape.size());

  const auto &smooth_scale = input_args[kInputIndex1];
  if (smooth_scale->GetType()->type_id() != kMetaTypeNone) {
    auto smooth_scale_shape = smooth_scale->GetShape()->GetShapeVector();
    MS_CHECK_VALUE(
      smooth_scale_shape.size() == 1,
      CheckAndConvertUtils::FormatCommMsg("The rank of smooth_scale must be 1, but got: ", smooth_scale_shape));

    if (!IsDynamic(smooth_scale_shape) && is_x_static) {
      MS_CHECK_VALUE(
        smooth_scale_shape[0] == (int64_t)(x_shape[x_shape.size() - 1] / kSplitNum),
        CheckAndConvertUtils::FormatCommMsg(
          "The dim of smooth_scale must equal to the last dim of x_shape divided by 2, but got smooth_scale_shape: ",
          smooth_scale_shape, ", x_shape: ", x_shape));
    }
  }

  MS_CHECK_VALUE(prim->HasAttr("dim"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + prim->name() + "', op must have attr 'dim'."));
  int64_t dim = GetValue<int64_t>(prim->GetAttr("dim"));
  MS_CHECK_VALUE(dim >= -x_rank && dim < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim, kIncludeLeft, {-x_rank, x_rank}, prim));
  if (dim < 0) {
    dim += x_rank;
  }

  BaseShapePtr out_y_shape_ptr;
  BaseShapePtr out_scale_shape_ptr;
  ShapeVector y_shape;
  if (IsDynamicRank(x_shape)) {
    out_scale_shape_ptr = x_shape_ptr;
  } else {
    MS_CHECK_VALUE(x_shape.size() > 1, CheckAndConvertUtils::FormatCommMsg(
                                         "The rank of input x must be greater than 1, but got: ", x_shape));
    x_shape[dim] = x_shape[dim] / kSplitNum;
    if (prim->HasAttr("HasReshape") && !IsDynamic(x_shape)) {
      auto input_element =
        std::accumulate(x_shape.begin(), x_shape.end() - 1, static_cast<int64_t>(1), std::multiplies<int64_t>());
      y_shape.push_back(input_element);
      y_shape.push_back(x_shape[dim]);
      out_y_shape_ptr = std::make_shared<abstract::Shape>(y_shape);
      out_scale_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector(y_shape.begin(), y_shape.end() - 1));
    } else {
      out_y_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
      out_scale_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector(x_shape.begin(), x_shape.end() - 1));
    }
  }

  std::vector<BaseShapePtr> shapes_list = {out_y_shape_ptr, out_scale_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr SwiGLUDynamicQuantFuncImpl::InferType(const PrimitivePtr &prim,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  auto smooth_scale_type = input_args[kInputIndex0]->GetType();

  auto quant_out_type = std::make_shared<TensorType>(kInt8);
  auto scale_out_type = std::make_shared<TensorType>(kFloat32);

  std::map<std::string, TypePtr> types = {{"x_type", x_type}, {"smooth_scale_type", smooth_scale_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  std::vector<TypePtr> types_list = {quant_out_type, scale_out_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace ops
}  // namespace mindspore
