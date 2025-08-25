/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "infer/deprecated_infer/infer_functions.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplScalarToArray(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list) {
  // Inputs: a scalar.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  AbstractScalarPtr arg = CheckArg<AbstractScalar>(op_name, args_abs_list, 0);
  return std::make_shared<AbstractTensor>(arg, std::make_shared<Shape>());
}

AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list) {
  // Inputs: a tensor with 0 shape.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto arg = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto a_shp = arg->shape();
  MS_EXCEPTION_IF_NULL(a_shp);
  if (!a_shp->shape().empty()) {
    MS_LOG(EXCEPTION) << "array_to_scalar requires zero size shape.";
  }
  return arg->element();
}

AbstractBasePtr InferImplBroadcastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_abs_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 2;
  CheckArgsSize(op_name, args_abs_list, args_size);
  auto xs = CheckArg<AbstractTuple>(op_name, args_abs_list, 0);
  auto ys = CheckArg<AbstractTuple>(op_name, args_abs_list, 1);
  auto x_value = xs->BuildValue();
  MS_EXCEPTION_IF_NULL(x_value);
  auto value_tuple_x = x_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_x);
  auto shp_tuple_x = value_tuple_x->value();
  ShapeVector shp_x;
  (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(shp_x),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  auto tupe_value_y = ys->BuildValue();
  MS_EXCEPTION_IF_NULL(tupe_value_y);
  auto value_tuple_y = tupe_value_y->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  ShapeVector shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });

  ShapeVector res = BroadcastShape(shp_x, shp_y);
  MS_EXCEPTION_IF_NULL(args_abs_list[1]);
  if (res.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_abs_list[0]->ToString() << "," << args_abs_list[1]->ToString();
  }

  AbstractBasePtrList elems;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(elems), [](int64_t n) -> AbstractBasePtr {
    return std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(n), kInt64);
  });
  return std::make_shared<AbstractTuple>(elems);
}

AbstractBasePtr InferImplMapCacheIdx(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 5;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  auto hash_map = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  MS_EXCEPTION_IF_NULL(hash_map->shape());

  auto indices = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  auto indices_shp = indices->shape();
  MS_EXCEPTION_IF_NULL(indices_shp);

  ShapeVector shape(indices_shp->shape().size(), -1);

  auto cache_idx = std::make_shared<AbstractTensor>(hash_map->element(), indices->shape());
  auto old_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));
  auto miss_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));
  auto swap_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));

  AbstractBasePtrList elements = {cache_idx, old_emb_idx, miss_emb_idx, swap_emb_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplCacheSwapTable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  auto cache_table = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto cache_table_shp = cache_table->shape();
  MS_EXCEPTION_IF_NULL(cache_table_shp);

  auto swap_cache_idx = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  auto swap_cache_idx_shp = swap_cache_idx->shape();
  MS_EXCEPTION_IF_NULL(swap_cache_idx_shp);

  auto cache_table_shape = cache_table_shp->shape();
  auto swap_cache_idx_shape = swap_cache_idx_shp->shape();
  ShapeVector shape;
  shape.emplace_back(swap_cache_idx_shape[0]);
  shape.emplace_back(cache_table_shape[1]);

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(cache_table->element(), std::make_shared<Shape>(shape));
  return ret;
}

AbstractBasePtr InferImplSubAndFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  auto input_x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto input_x_shp = input_x->shape();
  MS_EXCEPTION_IF_NULL(input_x_shp);

  ShapeVector shape(input_x_shp->shape().size(), -1);

  auto filter_res = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
  auto filter_idx = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
  AbstractBasePtrList elements = {filter_res, filter_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplRealInnerDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  if (out_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_abs_list[0]->ToString() << "," << args_abs_list[1]->ToString();
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list) {
  const std::string &op_name = primitive->name();
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto input_shp = input->shape()->shape();
  ValuePtr perm = primitive->GetAttr("perm");
  MS_EXCEPTION_IF_NULL(perm);
  auto perm_val = perm->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(perm_val);
  auto perm_val_data = perm_val->value();
  ShapeVector perm_vec;
  (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(perm_vec),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  ShapeVector result_shp;
  for (size_t i = 0; i < perm_vec.size(); i++) {
    auto idx = static_cast<size_t>(perm_vec[i]);
    result_shp.push_back(input_shp[idx]);
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp));
}

AbstractBasePtr InferImplMapUniform(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_abs_list) {
  // Inputs: one tensor.
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_abs_list, size_expected);
  return args_abs_list[0]->Broaden();
}

AbstractBasePtr InferImplSequenceMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list) {
  const std::string &op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_abs_list, size_expected);

  AbstractTensorPtr lengths = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  (void)CheckTensorDType(lengths, {kInt32, kInt64}, "Input 1 (lengths) for SequenceMask should be one of: %s");

  int64_t maxlen_value = 0;

  if (args_abs_list[1]->isa<AbstractScalar>()) {
    AbstractScalarPtr maxlen = CheckArg<AbstractScalar>(op_name, args_abs_list, 1);
    (void)CheckScalarType(maxlen, {kInt32, kInt64}, "Input 0 (maxlen) for SequenceMask should be one of: %s");

    TypePtr maxlen_type = nullptr;
    maxlen_type = maxlen->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(maxlen_type);

    if (maxlen_type->type_id() == TypeId::kNumberTypeInt32) {
      maxlen_value = static_cast<int64_t>(GetValue<int32_t>(maxlen->BuildValue()));
    } else if (maxlen_type->type_id() == TypeId::kNumberTypeInt64) {
      maxlen_value = GetValue<int64_t>(maxlen->BuildValue());
    }
  } else if (args_abs_list[1]->isa<AbstractTensor>()) {
    auto maxlen_tensor_ptr = args_abs_list[1]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor_ptr);
    auto maxlen_value_ptr = maxlen_tensor_ptr->BuildValue();
    MS_EXCEPTION_IF_NULL(maxlen_value_ptr);
    auto maxlen_tensor = maxlen_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor);
    auto maxlen_tensor_cpu = maxlen_tensor->cpu();
    maxlen_value = *static_cast<int64_t *>(maxlen_tensor_cpu->data_c());
  }

  if (maxlen_value <= 0) {
    MS_LOG(EXCEPTION) << "maxlen must be positive, but got: " << maxlen_value;
  }

  ShapeVector lengths_shape = lengths->shape()->shape();
  lengths_shape.push_back(maxlen_value);
  ShapePtr output_shape = std::make_shared<Shape>(lengths_shape);
  return std::make_shared<AbstractTensor>(kBool, output_shape);
}
}  // namespace abstract
}  // namespace mindspore
