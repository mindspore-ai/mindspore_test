/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "include/common/utils/parallel_context.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>

namespace mindspore::parallel {
namespace {
std::vector<std::string> kParallelModeList = {kStandalone, kDataParallel, kHybridParallel, kSemiAutoParallel,
                                              kAutoParallel};
std::vector<std::string> kStrategySearchModeList = {kDynamicProgramming, kRecursiveProgramming, kShardingPropagation};

std::vector<std::string> kCommuniParallelModeList = {kAllGroupParallel, kSameServerGroupParallel, kNoGroupParallel};

std::vector<std::string> kFusionModeList = {kFusionAuto, kFusionSize, kFusionIndex};
}  // namespace

std::shared_ptr<ParallelContext> ParallelContext::GetInstance() {
  static std::shared_ptr<ParallelContext> inst_context_ = std::shared_ptr<ParallelContext>(new ParallelContext());
  return inst_context_;
}

ParallelContext::ParallelContext() { Reset(); }

void ParallelContext::Reset() {
  gradients_mean_ = false;
  full_batch_ = false;
  full_batch_is_set_ = false;
  gradient_fp32_sync_ = true;
  loss_repeated_mean_ = true;
  device_num_ = 1;
  global_rank_ = 0;
  device_num_is_set_ = false;
  global_rank_is_set_ = false;
  parallel_mode_ = kStandalone;
  parameter_broadcast_ = false;
  parameter_broadcast_is_set_ = false;
  enable_all_reduce_fusion_ = false;
  enable_all_gather_fusion_ = false;
  enable_reduce_scatter_fusion_ = false;
  strategy_json_config_file_type_ = "";
  strategy_json_config_file_path_ = "";
  strategy_json_config_file_mode_ = "";
  strategy_ckpt_load_file_ = "";
  strategy_ckpt_save_file_ = "";
  enable_parallel_optimizer_ = false;
  force_fp32_communication_ = false;
  all_reduce_fusion_split_indices_.clear();
  all_reduce_fusion_split_sizes_.clear();
  strategy_search_mode_ = kRecursiveProgramming;
  pipeline_stage_split_num_ = 1;
  pipeline_segment_split_num_ = 1;
  grad_accumulation_step_ = 1;
  communi_parallel_mode_ = kAllGroupParallel;
  optimizer_weight_shard_size_ = -1;
  optimizer_weight_shard_aggregated_save_ = false;
  enable_all2all_ = false;
  grad_accumulation_shard_ = false;
  parallel_optimizer_threshold_ = -1;
  sharding_propagation_ = false;
  dataset_strategy_.clear();
  dp_fusion_threshold_mb_ = kDataParallelFusionThreshold;
  fusion_threshold_mb_ = kFusionThreshold;
  allgather_fusion_threshold_mb_ = kFusionThreshold;
  reducescatter_fusion_threshold_mb_ = kFusionThreshold;
  fusion_threshold_is_set_ = true;
  fusion_mode_ = kFusionAuto;
  group_ckpt_save_file_ = "";
  pipeline_micro_size_ = 1;
  dataset_repeat_dim_right_ = false;
  hccl_test_available_ = false;
  enable_micro_interleaved_ = false;
  enable_fine_grained_micro_interleaved_ = false;
  do_transform_ = false;
  direct_split_ = false;
  pipeline_result_broadcast_ = false;
  stra_file_only_trainable_params_ = true;
  pipeline_interleave_ = false;
  pipeline_scheduler_ = kPipeline1F1B;
  auto_pipeline_ = false;
  dump_local_norm_ = false;
  dump_local_norm_path_ = "";
  dump_device_local_norm_ = false;
  is_dynamic_shape_parallel_ = false;
  dynamic_shape_parallel_flag_is_set_ = false;
  init_param_in_compile_ = true;
  auto_parallel_new_interface_ = false;
  dataset_strategy_devmat_.clear();
  dataset_strategy_tensormap_.clear();
  dataset_strategy_alias_name_.clear();
  zero3_ = false;
}

void ParallelContext::set_device_num(int64_t device_num) {
  device_num_ = device_num;
  device_num_is_set_ = true;
}

void ParallelContext::set_is_dynamic_shape_parallel(bool flag) {
  is_dynamic_shape_parallel_ = flag;
  dynamic_shape_parallel_flag_is_set_ = true;
}

void ParallelContext::set_fusion_threshold_mb(int64_t fusion_threshold) {
  fusion_threshold_mb_ = fusion_threshold;
  dp_fusion_threshold_mb_ = fusion_threshold;
  fusion_threshold_is_set_ = true;
  enable_all_reduce_fusion_ = true;
}

void ParallelContext::set_allgather_fusion_threshold_mb(int64_t fusion_threshold) {
  allgather_fusion_threshold_mb_ = fusion_threshold;
  enable_all_gather_fusion_ = true;
}

void ParallelContext::set_reducescatter_fusion_threshold_mb(int64_t fusion_threshold) {
  reducescatter_fusion_threshold_mb_ = fusion_threshold;
  enable_reduce_scatter_fusion_ = true;
}

bool ParallelContext::set_fusion_mode(const std::string &fusion_mode) {
  auto iter = std::find(kFusionModeList.begin(), kFusionModeList.end(), fusion_mode);
  if (iter == kFusionModeList.end()) {
    MS_LOG(INFO) << "Invalid fusion mode:" << fusion_mode;
    return false;
  }
  fusion_mode_ = fusion_mode;
  return true;
}

void ParallelContext::set_global_rank(int64_t global_rank) {
  global_rank_ = global_rank;
  global_rank_is_set_ = true;
}

void ParallelContext::set_gradients_mean(bool gradients_mean) { gradients_mean_ = gradients_mean; }

void ParallelContext::set_full_batch(bool full_batch) {
  full_batch_ = full_batch;
  full_batch_is_set_ = true;
}

void ParallelContext::set_dataset_strategy(const std::vector<std::vector<int64_t>> &dataset_strategy) {
  dataset_strategy_ = dataset_strategy;
}

void ParallelContext::set_dataset_layout(
  const std::vector<std::vector<int64_t>> &dataset_strategy_devmat,
  const std::vector<std::vector<std::vector<int64_t>>> &dataset_strategy_tensormap,
  const std::vector<std::vector<std::string>> &dataset_strategy_alias_name) {
  dataset_strategy_devmat_ = dataset_strategy_devmat;
  dataset_strategy_tensormap_ = dataset_strategy_tensormap;
  dataset_strategy_alias_name_ = dataset_strategy_alias_name;
}

void ParallelContext::set_grad_accumulation_step(int64_t grad_accumulation_step) {
  grad_accumulation_step_ = grad_accumulation_step;
}

void ParallelContext::set_gradient_fp32_sync(bool gradient_fp32_sync) { gradient_fp32_sync_ = gradient_fp32_sync; }

void ParallelContext::set_loss_repeated_mean(bool loss_repeated_mean) { loss_repeated_mean_ = loss_repeated_mean; }

void ParallelContext::set_pipeline_stage_split_num(const int64_t stage_num) { pipeline_stage_split_num_ = stage_num; }

void ParallelContext::set_pipeline_interleave(const bool pipeline_interleave) {
  pipeline_interleave_ = pipeline_interleave;
}

void ParallelContext::set_pipeline_interleave_temp(const bool pipeline_interleave_temp) {
  pipeline_interleave_temp_ = pipeline_interleave_temp;
}

void ParallelContext::set_pipeline_scheduler(const std::string &pipeline_scheduler) {
  pipeline_scheduler_ = pipeline_scheduler;
}

void ParallelContext::set_pipeline_scheduler_temp(const std::string &pipeline_scheduler_temp) {
  pipeline_scheduler_temp_ = pipeline_scheduler_temp;
}

void ParallelContext::set_pipeline_segment_split_num(const int64_t segment_num) {
  pipeline_segment_split_num_ = segment_num;
}

bool ParallelContext::set_parallel_mode(const std::string &parallel_mode) {
  auto iter = std::find(kParallelModeList.begin(), kParallelModeList.end(), parallel_mode);
  if (iter == kParallelModeList.end()) {
    MS_LOG(INFO) << "Invalid parallel mode:" << parallel_mode;
    return false;
  }
  parallel_mode_ = parallel_mode;
  return true;
}

bool ParallelContext::set_strategy_search_mode(const std::string &strategy_search_mode) {
  auto iter = std::find(kStrategySearchModeList.begin(), kStrategySearchModeList.end(), strategy_search_mode);
  if (iter == kStrategySearchModeList.end()) {
    MS_LOG(INFO) << "Invalid strategy search mode mode: " << strategy_search_mode;
    return false;
  }
  strategy_search_mode_ = strategy_search_mode;
  return true;
}

void ParallelContext::set_parameter_broadcast(bool parameter_broadcast) {
  parameter_broadcast_ = parameter_broadcast;
  parameter_broadcast_is_set_ = true;
}

void ParallelContext::set_ops_strategy_json_config(const std::string &type, const std::string &path,
                                                   const std::string &mode) {
  strategy_json_config_file_type_ = type;
  strategy_json_config_file_path_ = path;
  strategy_json_config_file_mode_ = mode;
}

void ParallelContext::set_strategy_ckpt_load_file(const std::string &strategy_ckpt_load_file) {
  strategy_ckpt_load_file_ = strategy_ckpt_load_file;
}

void ParallelContext::set_strategy_ckpt_save_file(const std::string &strategy_ckpt_save_file) {
  strategy_ckpt_save_file_ = strategy_ckpt_save_file;
}

void ParallelContext::set_group_ckpt_save_file(const std::string &group_ckpt_save_file) {
  group_ckpt_save_file_ = group_ckpt_save_file;
}

void ParallelContext::set_optimizer_weight_shard_size(int64_t optimizer_weight_shard_size) {
  optimizer_weight_shard_size_ = optimizer_weight_shard_size;
}

void ParallelContext::set_optimizer_weight_shard_aggregated_save(bool optimizer_weight_shard_aggregated_save) {
  optimizer_weight_shard_aggregated_save_ = optimizer_weight_shard_aggregated_save;
}

void ParallelContext::SetAllReduceFusionSplitIndices(const std::vector<uint32_t> &indices, const std::string &group) {
  if (!group.empty() && group.find(TypeIdLabel(kNumberTypeFloat)) == std::string::npos &&
      group.find(TypeIdLabel(kNumberTypeFloat16)) == std::string::npos &&
      group.find(TypeIdLabel(kNumberTypeFloat32)) == std::string::npos) {
    all_reduce_fusion_split_indices_[group + TypeIdLabel(kNumberTypeFloat)] = indices;
    all_reduce_fusion_split_indices_[group + TypeIdLabel(kNumberTypeFloat16)] = indices;
    all_reduce_fusion_split_indices_[group + TypeIdLabel(kNumberTypeFloat32)] = indices;
  }
  all_reduce_fusion_split_indices_[group] = indices;
  enable_all_reduce_fusion_ = true;
}

std::vector<uint32_t> ParallelContext::GetAllReduceFusionSplitIndices(const std::string &group) const {
  auto iter = all_reduce_fusion_split_indices_.find(group);
  if (iter != all_reduce_fusion_split_indices_.end()) {
    return iter->second;
  }
  return {};
}

void ParallelContext::SetAllReduceFusionSplitSizes(const std::vector<uint32_t> &sizes, const std::string &group) {
  if (!group.empty() && group.find(TypeIdLabel(kNumberTypeFloat)) == std::string::npos &&
      group.find(TypeIdLabel(kNumberTypeFloat16)) == std::string::npos &&
      group.find(TypeIdLabel(kNumberTypeFloat32)) == std::string::npos) {
    all_reduce_fusion_split_sizes_[group + TypeIdLabel(kNumberTypeFloat)] = sizes;
    all_reduce_fusion_split_sizes_[group + TypeIdLabel(kNumberTypeFloat16)] = sizes;
    all_reduce_fusion_split_sizes_[group + TypeIdLabel(kNumberTypeFloat32)] = sizes;
  }
  all_reduce_fusion_split_sizes_[group] = sizes;
}

std::vector<uint32_t> ParallelContext::GetAllReduceFusionSplitSizes(const std::string &group) const {
  auto iter = all_reduce_fusion_split_sizes_.find(group);
  if (iter != all_reduce_fusion_split_sizes_.end()) {
    return iter->second;
  }
  return {};
}

bool ParallelContext::set_communi_parallel_mode(const std::string &communi_parallel_mode) {
  auto iter = std::find(kCommuniParallelModeList.begin(), kCommuniParallelModeList.end(), communi_parallel_mode);
  if (iter == kCommuniParallelModeList.end()) {
    MS_LOG(INFO) << "Invalid communication parallel mode:" << communi_parallel_mode;
    return false;
  }

  communi_parallel_mode_ = communi_parallel_mode;
  return true;
}

// Restore the parameters' shape for evaluation/prediction in auto-parallel or semi-auto-parallel mode
void ParallelContext::ParallelParameterContextRestoreShape(const FuncGraphPtr &func_graph,
                                                           const ParameterPtr &param_node,
                                                           const AbstractBasePtr &ptr) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ptr);
  if (!ParallelContextCareGraph(func_graph)) {
    return;
  }

  auto param_info = param_node->param_info();
  if (!param_info) {
    return;
  }
  auto shape = param_info->parameter_shape();
  if (shape.empty()) {
    MS_LOG(INFO) << "The parameter " << param_node->name() << "'s parameter_shape in param_info is empty";
    return;
  }
  std::shared_ptr<abstract::BaseShape> base_shape = std::make_shared<abstract::Shape>(shape);
  ptr->set_shape(base_shape);
  MS_LOG(INFO) << "The parameter name is " << param_node->name() << ", the shape is " << shape;
}

bool ParallelContext::ParallelContextCareGraph(const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(kSkipAutoParallelCompile)) {
    return false;
  }

  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kAutoParallel && parallel_mode != kSemiAutoParallel) {
    return false;
  }

  return true;
}

void ParallelContext::set_enable_all2all(const bool enable) { enable_all2all_ = enable; }

void ParallelContext::set_enable_micro_interleaved(const bool enable_micro_interleaved) {
  enable_micro_interleaved_ = enable_micro_interleaved;
}

void ParallelContext::set_enable_fine_grained_micro_interleaved(const bool enable_fine_grained_micro_interleaved) {
  enable_fine_grained_micro_interleaved_ = enable_fine_grained_micro_interleaved;
}

void ParallelContext::set_pipeline_micro_size(const size_t pipeline_micro_size) {
  pipeline_micro_size_ = pipeline_micro_size;
}

void ParallelContext::set_auto_pipeline(const bool auto_pipeline) { auto_pipeline_ = auto_pipeline; }

void ParallelContext::set_do_transform(const bool do_transform) { do_transform_ = do_transform; }

void ParallelContext::set_stra_file_only_trainable_params(const bool stra_file_only_trainable_params) {
  stra_file_only_trainable_params_ = stra_file_only_trainable_params;
}
void ParallelContext::set_zero3(const bool zero3) { zero3_ = zero3; }

void ParallelContext::set_sharding_propagation(const bool stra_pto) { sharding_propagation_ = stra_pto; }

void ParallelContext::set_dump_local_norm(const bool dump_ln) { dump_local_norm_ = dump_ln; }

void ParallelContext::set_dump_local_norm_path(const std::string &dump_local_norm_path) {
  dump_local_norm_path_ = dump_local_norm_path;
}

void ParallelContext::set_dump_device_local_norm(const bool dump_device_local_norm) {
  dump_device_local_norm_ = dump_device_local_norm;
}

void ParallelContext::set_init_param_in_compile(const bool init_param_in_compile) {
  init_param_in_compile_ = init_param_in_compile;
}

void ParallelContext::set_auto_parallel_new_interface(const bool auto_parallel_new_interface) {
  auto_parallel_new_interface_ = auto_parallel_new_interface;
}

}  // namespace mindspore::parallel
