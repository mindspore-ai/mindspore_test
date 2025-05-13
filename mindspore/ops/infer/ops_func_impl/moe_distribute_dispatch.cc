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

#include <set>
#include <memory>
#include <algorithm>

#include "infer/ops_func_impl/moe_distribute_dispatch.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kXDim = 2;
constexpr size_t kX = 0;
constexpr size_t kExpertIds = 1;
constexpr size_t kEpWorldSize = 2;
constexpr size_t kEpRankId = 3;
constexpr size_t kMoeExpertNum = 4;
constexpr size_t kTpWorldSize = 10;
constexpr size_t kSharedExpertNum = 13;
constexpr size_t kSharedExpertRankNum = 14;
constexpr size_t kQuantMode = 15;
constexpr size_t kGlobalBs = 16;
}  // namespace

BaseShapePtr MoeDistributeDispatchFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto x_shape_ptr = input_args[kX]->GetShape();
  const auto &x_shape = x_shape_ptr->GetShapeVector();
  auto expert_ids_shape_ptr = input_args[kExpertIds]->GetShape();
  const auto &expert_ids_shape = expert_ids_shape_ptr->GetShapeVector();

  if (IsDynamicRank(x_shape) || IsDynamicRank(expert_ids_shape)) {
    ShapeVector dyshape_2d_vec{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
    auto dyshape_2d = std::make_shared<abstract::TensorShape>(dyshape_2d_vec);
    ShapeVector dyshape_1d_vec{abstract::TensorShape::kShapeDimAny};
    auto dyshape_1d = std::make_shared<abstract::TensorShape>(dyshape_1d_vec);

    std::vector<BaseShapePtr> shapes_list;
    shapes_list = {dyshape_2d, dyshape_1d, dyshape_1d, dyshape_1d, dyshape_1d, dyshape_1d, dyshape_1d};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  MS_CHECK_VALUE(x_shape.size() == kXDim, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                            "rank of x", SizeToLong(x_shape.size()), kEqual, kXDim, primitive));
  MS_CHECK_VALUE(expert_ids_shape.size() == kXDim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of expert_ids", SizeToLong(expert_ids_shape.size()),
                                                             kEqual, kXDim, primitive));

  auto BS = x_shape[kDim0];
  auto H = x_shape[kDim1];
  auto K = expert_ids_shape[kDim1];
  auto ep_world_size = GetScalarValue<int64_t>(input_args[kEpWorldSize]->GetValue()).value();
  auto ep_rank_id = GetScalarValue<int64_t>(input_args[kEpRankId]->GetValue()).value();
  auto moe_expert_num = GetScalarValue<int64_t>(input_args[kMoeExpertNum]->GetValue()).value();
  auto tp_world_size_scalar = GetScalarValue<int64_t>(input_args[kTpWorldSize]->GetValue());
  auto tp_world_size = tp_world_size_scalar.has_value() ? tp_world_size_scalar.value() : 1;
  tp_world_size = (tp_world_size == 0) ? 1 : tp_world_size;
  auto shared_expert_num_scalar = GetScalarValue<int64_t>(input_args[kSharedExpertNum]->GetValue());
  auto shared_expert_num = shared_expert_num_scalar.has_value() ? shared_expert_num_scalar.value() : 0;
  auto shared_expert_rank_num_scalar = GetScalarValue<int64_t>(input_args[kSharedExpertRankNum]->GetValue());
  auto shared_expert_rank_num = shared_expert_rank_num_scalar.has_value() ? shared_expert_rank_num_scalar.value() : 0;
  auto global_bs_scalar = GetScalarValue<int64_t>(input_args[kGlobalBs]->GetValue());
  auto global_bs = global_bs_scalar.has_value() ? global_bs_scalar.value() : 0;
  global_bs = (global_bs == 0) ? BS * ep_world_size : global_bs;

  int64_t local_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
  if (ep_rank_id < shared_expert_rank_num) {
    local_expert_num = 1;
  }

  int64_t A = 0;
  int64_t expand_idx_dim = BS * K;
  if (BS == abstract::Shape::kShapeDimAny) {
    A = abstract::Shape::kShapeDimAny;
    expand_idx_dim = abstract::Shape::kShapeDimAny;
  } else {
    if (ep_rank_id < shared_expert_rank_num) {
      A = BS * ep_world_size * shared_expert_num / shared_expert_rank_num;
    } else {
      A = global_bs * std::min(local_expert_num, K);
    }
    A *= tp_world_size;
  }

  int64_t ep_recv_counts_dim = abstract::Shape::kShapeDimAny;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == kAscendVersion910b) {
    if (BS != abstract::Shape::kShapeDimAny && K != abstract::Shape::kShapeDimAny) {
      constexpr int64_t one_node_cards = 8;
      constexpr int64_t factor = 2;
      int64_t server_num = ep_world_size / one_node_cards;
      ep_recv_counts_dim = moe_expert_num + factor * global_bs * K * server_num;
    }
  } else if (soc_version == kAscendVersion910_93) {
    ep_recv_counts_dim = ep_world_size * tp_world_size * local_expert_num;
  } else {
    MS_LOG(EXCEPTION) << "'MoeDistributeDispatch' only support [" << kAscendVersion910b << ", " << kAscendVersion910_93
                      << "], but got " << soc_version;
  }

  ShapeVector expand_x_shape = {A, H};
  ShapeVector dynamic_scales_shape = {A};
  ShapeVector expand_idx_shape = {expand_idx_dim};
  ShapeVector expert_token_nums_shape = {local_expert_num};
  ShapeVector ep_recv_counts_shape = {ep_recv_counts_dim};
  ShapeVector tp_recv_counts_shape = {tp_world_size};
  ShapeVector expand_scales_shape = {A};

  std::vector<BaseShapePtr> shapes_list;
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expand_x_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(dynamic_scales_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expand_idx_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expert_token_nums_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(ep_recv_counts_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(tp_recv_counts_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expand_scales_shape));

  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr MoeDistributeDispatchFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat16, kBFloat16};
  const auto &x_type = input_args[kX]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, tensor_valid_types, prim_name);

  auto quant_mode_scalar = GetScalarValue<int64_t>(input_args[kQuantMode]->GetValue());
  auto quant_mode = quant_mode_scalar.has_value() ? quant_mode_scalar.value() : 0;
  auto expand_x_type = input_args[kX]->GetType();
  if (quant_mode != 0) {
    expand_x_type = std::make_shared<TensorType>(kInt8);
  }

  const auto &dynamic_scales_type = std::make_shared<TensorType>(kFloat32);
  const auto &expand_idx_type = std::make_shared<TensorType>(kInt32);
  const auto &expert_token_nums_type = std::make_shared<TensorType>(kInt64);
  const auto &ep_recv_counts_type = std::make_shared<TensorType>(kInt32);
  const auto &tp_recv_counts_type = std::make_shared<TensorType>(kInt32);
  const auto &expand_scales_type = std::make_shared<TensorType>(kFloat32);

  std::vector<TypePtr> types_list;
  types_list = {expand_x_type,       dynamic_scales_type, expand_idx_type,   expert_token_nums_type,
                ep_recv_counts_type, tp_recv_counts_type, expand_scales_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace ops
}  // namespace mindspore
