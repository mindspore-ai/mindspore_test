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

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/moe_distribute_dispatch.h"

namespace mindspore {
namespace ops {
namespace {
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

void MoeDistributeDispatchFuncImpl::CheckInputs(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kX];
  auto &expert_ids_info = input_infos[kExpertIds];
  auto any_dim = abstract::Shape::kShapeDimAny;

  if (x_info->IsNone() || expert_ids_info->IsNone()) {
    MS_EXCEPTION(ShapeError) << "For op[" << op_name << "], the input x or input expert_ids should have real "
                             << "shape, but get None!";
    return;
  }

  CheckRank(x_info, kInputRankSize, op_name, "x");
  CheckRank(expert_ids_info, kInputRankSize, op_name, "expert_ids");
  const auto &x_shp = x_info->GetShape();
  const auto &expert_ids_shp = expert_ids_info->GetShape();
  bool is_dynamic_shape = (x_shp.front() == any_dim) || (expert_ids_shp.front() == any_dim);
  if (is_dynamic_shape && (x_shp.front() != expert_ids_shp.front())) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the first dim of x and expert_ids must be the same.";
  }
}

ShapeArray MoeDistributeDispatchFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  // Get input tensor shape.
  auto &x_info = input_infos[kX];
  auto &expert_ids_info = input_infos[kExpertIds];

  auto any_dim = abstract::Shape::kShapeDimAny;
  auto any_shape = abstract::TensorShape::kShapeRankAny;
  if (x_info->IsDynamicRank() || expert_ids_info->IsDynamicRank()) {
    return {{any_shape}, {any_dim}, {any_dim}, {any_dim}, {any_dim}, {any_dim}, {any_dim}};
  }
  CheckInputs(primitive, input_infos);

  const auto &x_shape = x_info->GetShape();
  const auto &expert_ids_shape = expert_ids_info->GetShape();
  auto BS = x_shape[kDim0];
  auto H = x_shape[kDim1];
  auto K = expert_ids_shape[kDim1];
  auto ep_world_size = input_infos[kEpWorldSize]->GetScalarValue<int64_t>().value();
  auto ep_rank_id = input_infos[kEpRankId]->GetScalarValue<int64_t>().value();
  auto moe_expert_num = input_infos[kMoeExpertNum]->GetScalarValue<int64_t>().value();
  auto tp_world_size_scalar = input_infos[kTpWorldSize]->GetScalarValue<int64_t>();
  auto tp_world_size = tp_world_size_scalar.has_value() ? tp_world_size_scalar.value() : 1;
  tp_world_size = (tp_world_size == 0) ? 1 : tp_world_size;
  auto shared_expert_num_scalar = input_infos[kSharedExpertNum]->GetScalarValue<int64_t>();
  auto shared_expert_num = shared_expert_num_scalar.has_value() ? shared_expert_num_scalar.value() : 0;
  auto shared_expert_rank_num_scalar = input_infos[kSharedExpertRankNum]->GetScalarValue<int64_t>();
  auto shared_expert_rank_num = shared_expert_rank_num_scalar.has_value() ? shared_expert_rank_num_scalar.value() : 0;
  auto global_bs_scalar = input_infos[kGlobalBs]->GetScalarValue<int64_t>();
  auto global_bs = global_bs_scalar.has_value() ? global_bs_scalar.value() : 0;
  global_bs = (global_bs == 0) ? BS * ep_world_size : global_bs;

  int64_t local_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
  if (ep_rank_id < shared_expert_rank_num) {
    local_expert_num = 1;
  }

  int64_t A = any_dim;
  int64_t expand_x_dim = any_dim;
  int64_t expand_idx_dim = any_dim;
  if (BS != any_dim) {
    if (ep_rank_id < shared_expert_rank_num) {
      A = BS * ep_world_size * shared_expert_num / shared_expert_rank_num;
    } else {
      A = global_bs * std::min(local_expert_num, K);
    }
    expand_x_dim = A * tp_world_size;
    if (K != any_dim) {
      expand_idx_dim = BS * K;
    }
  }

  int64_t ep_recv_counts_dim = any_dim;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == kAscendVersion910b) {
    if (BS != any_dim && K != any_dim) {
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

  ShapeVector expand_x_shape = {expand_x_dim, H};
  ShapeVector dynamic_scales_shape = {A};
  ShapeVector expand_idx_shape = {expand_idx_dim};
  ShapeVector expert_token_nums_shape = {local_expert_num};
  ShapeVector ep_recv_counts_shape = {ep_recv_counts_dim};
  ShapeVector tp_recv_counts_shape = {tp_world_size};
  ShapeVector expand_scales_shape = {A};

  ShapeArray shapes_list;
  (void)shapes_list.emplace_back(expand_x_shape);
  (void)shapes_list.emplace_back(dynamic_scales_shape);
  (void)shapes_list.emplace_back(expand_idx_shape);
  (void)shapes_list.emplace_back(expert_token_nums_shape);
  (void)shapes_list.emplace_back(ep_recv_counts_shape);
  (void)shapes_list.emplace_back(tp_recv_counts_shape);
  (void)shapes_list.emplace_back(expand_scales_shape);

  return shapes_list;
}

TypeIdList MoeDistributeDispatchFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const std::set<TypeId> x_valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  const std::set<TypeId> ids_valid_types = {kNumberTypeInt32};
  auto x_type = input_infos[kX]->GetType();
  auto ids_type = input_infos[kExpertIds]->GetType();
  CheckType(x_valid_types, x_type, op_name, "x");
  CheckType(ids_valid_types, ids_type, op_name, "expert_ids");

  auto quant_mode_scalar = input_infos[kQuantMode]->GetScalarValue<int64_t>();
  auto quant_mode = quant_mode_scalar.has_value() ? quant_mode_scalar.value() : 0;
  auto expand_x_type = x_type;
  if (quant_mode != 0) {
    expand_x_type = kNumberTypeInt8;
  }

  auto dynamic_scales_type = kNumberTypeFloat32;
  auto expand_idx_type = kNumberTypeInt32;
  auto expert_token_nums_type = kNumberTypeInt64;
  auto ep_recv_counts_type = kNumberTypeInt32;
  auto tp_recv_counts_type = kNumberTypeInt32;
  auto expand_scales_type = kNumberTypeFloat32;

  TypeIdList types_list;
  types_list = {expand_x_type,       dynamic_scales_type, expand_idx_type,   expert_token_nums_type,
                ep_recv_counts_type, tp_recv_counts_type, expand_scales_type};
  return types_list;
}
}  // namespace ops
}  // namespace mindspore
