/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_COSTMODEL_CONTEXT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_COSTMODEL_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/common/visible.h"

namespace mindspore {
namespace parallel {
constexpr char OPERATOR_TO_OPERATOR_CONNECTOR[] = "-";
constexpr float DEFAULT_DEVICE_MEMORY_CAPACITY = (1024.0 * 1024.0 * 1024.0 * 16.0);
constexpr float DEFAULT_COST_MODEL_ALPHA = 1.0;
constexpr float DEFAULT_COST_MODEL_BETA_ASCEND = 400.0;  // for 'device_target = Ascend'
constexpr float DEFAULT_COST_MODEL_BETA_GPU = 50.0;      // for 'device_target = GPU'
constexpr float DEFAULT_COST_MODEL_GAMMA = 0.001;
#define DEFAULT_COST_MODEL_SIMPLIFY_CALCULATION true
constexpr float DEFAULT_COST_MODEL_COMMUNI_THRESHOLD = 2048.0;
constexpr float DEFAULT_COST_MODEL_COMMUNI_CONST = 3072.0;
constexpr float DEFAULT_COST_MODEL_COMMUNI_BIAS = 1024.0;
#define DEFAULT_TENSOR_SLICE_ALIGNMENT_ENABLE false
constexpr size_t DEFAULT_TENSOR_SLICE_ALIGNMENT_SIZE = 16;
#define DEFAULT_FULLY_USE_DEVICES false
#define DEFAULT_ELEMENTWISE_OP_STRA_FOLLOW false
#define DEFAULT_IS_MULTI_SUBGRAPHS false
#define INFERENCE_PHASE 1
#define DEFAULT_TRIANGLE_STAR_STRATEGY_OVERWRITE true
#define DEFAULT_DP_ALGO_ENABLE_APPROX false
constexpr float DEFAULT_DP_ALGO_APPROX_EPSILON = 0.1;
constexpr float DEFAULT_RP_MATMUL_MEM_COEF = 0.1;
#define DEFAULT_DP_ALGO_SINGLE_LOOP false
constexpr int64_t TRAINING_PHASE = 0;

class FRONTEND_EXPORT CostModelContext {
 public:
  ~CostModelContext() = default;
  CostModelContext(const CostModelContext &) = delete;
  CostModelContext &operator=(const CostModelContext &) = delete;
  void ResetCostModel();
  void ResetAlgoParameters();

  static std::shared_ptr<CostModelContext> GetInstance();
  void PrintCostModel();

  void set_costmodel_context_for_device(const std::string &device_target);
  // DEVICE_MEMORY_CAPACITY
  void set_device_memory_capacity(double dm_capacity);
  double device_memory_capacity() const { return device_memory_capacity_; }

  // COST_MODEL_ALPHA
  void set_costmodel_alpha(double cm_alpha);
  double costmodel_alpha() const { return costmodel_alpha_; }

  // COST_MODEL_BETA
  void set_costmodel_beta(double cm_beta);
  double costmodel_beta() const { return costmodel_beta_; }

  // COST_MODEL_GAMMA
  void set_costmodel_gamma(double cm_gamma);
  double costmodel_gamma() const { return costmodel_gamma_; }

  // COST_MODEL_SIMPLIFY_CALCULATION
  void set_costmodel_simplify_cal(bool cm_simplify);
  bool costmodel_simplify_cal() const { return costmodel_simplify_cal_; }

  // COST_MODEL_COMMUNI_THRESHOLD
  void set_costmodel_communi_threshold(double cm_communi_th);
  double costmodel_communi_threshold() const { return costmodel_communi_threshold_; }

  // COST_MODEL_COMMUNI_CONST
  void set_costmodel_communi_const(double cm_communi_const);
  double costmodel_communi_const() const { return costmodel_communi_const_; }

  // COST_MODEL_COMMUNI_BIAS
  void set_costmodel_communi_bias(double cm_communi_bias);
  double costmodel_communi_bias() const { return costmodel_communi_bias_; }

  void set_multi_subgraphs(bool multi_graphs);
  bool is_multi_subgraphs() const { return is_multi_subgraphs_; }

  void set_costmodel_allreduce_fusion_algorithm(int64_t algorithm);
  int64_t costmodel_allreduce_fusion_algorithm() const { return costmodel_allreduce_fusion_algorithm_; }

  void set_costmodel_allreduce_fusion_times(int64_t allreduce_fusion_times);
  int64_t costmodel_allreduce_fusion_times() const { return costmodel_allreduce_fusion_times_; }

  void set_costmodel_allreduce_fusion_tail_percent(double tail_percent);
  double costmodel_allreduce_fusion_tail_percent() const { return costmodel_allreduce_fusion_tail_percent_; }

  void set_costmodel_allreduce_fusion_tail_time(double tail_time);
  double costmodel_allreduce_fusion_tail_time() const { return costmodel_allreduce_fusion_tail_time_; }

  void set_costmodel_allreduce_fusion_allreduce_inherent_time(double allreduce_inherent_time);
  double costmodel_allreduce_fusion_allreduce_inherent_time() const {
    return costmodel_allreduce_fusion_allreduce_inherent_time_;
  }

  void set_costmodel_allreduce_fusion_allreduce_bandwidth(double allreduce_bandwidth);
  double costmodel_allreduce_fusion_allreduce_bandwidth() const {
    return costmodel_allreduce_fusion_allreduce_bandwidth_;
  }

  void set_costmodel_allreduce_fusion_computation_time_parameter(double computation_time_parameter);
  double costmodel_allreduce_fusion_computation_time_parameter() const {
    return costmodel_allreduce_fusion_computation_time_parameter_;
  }

  // TENSOR_SLICE_ALIGNMENT_ENABLE
  void set_tensor_slice_alignment_enable(bool ts_align);
  bool tensor_slice_alignment_enable() const { return tensor_slice_alignment_enable_; }

  // TENSOR_SLICE_ALIGNMENT_SIZE
  void set_tensor_slice_alignment_size(size_t ts_align_size);
  size_t tensor_slice_alignment_size() const { return tensor_slice_alignment_size_; }

  // FULLY_USE_DEVICES
  void set_fully_use_device(bool fully_use);
  bool fully_use_device() const { return fully_use_device_; }

  // ELEMENTWISE_OP_STRA_FOLLOW
  void set_elementwise_stra_follow(bool elementwise_follow);
  bool elementwise_stra_follow() const { return elementwise_stra_follow_; }

  void set_triangle_star_strategy_overwrite(bool overwrite);
  bool triangle_star_strategy_overwrite() const { return triangle_star_strategy_overwrite_; }

  void set_run_phase(int64_t phase);
  int64_t run_phase() const { return run_phase_; }

  void set_dp_algo_approxi_epsilon(double epsilon);
  double dp_algo_approxi_epsilon() const { return dp_algo_approxi_epsilon_; }

  void set_rp_matmul_mem_coef(double coef);
  double rp_matmul_mem_coef() const { return rp_matmul_mem_coef_; }

  void set_dp_algo_enable_approxi(bool approxi);
  bool dp_algo_enable_approxi() const { return dp_algo_enable_approxi_; }

  void set_dp_algo_single_loop(bool single_loop);
  bool dp_algo_single_loop() const { return dp_algo_single_loop_; }

 private:
  CostModelContext();
  static std::shared_ptr<CostModelContext> cm_context_inst_;

  // DEVICE_MEMORY_CAPACITY
  double device_memory_capacity_;

  // COST_MODEL_ALPHA
  double costmodel_alpha_;

  // COST_MODEL_BETA
  double costmodel_beta_;

  // COST_MODEL_GAMMA
  double costmodel_gamma_;

  // COST_MODEL_SIMPLIFY_CALCULATION
  bool costmodel_simplify_cal_;

  // COST_MODEL_COMMUNI_THRESHOLD
  double costmodel_communi_threshold_;

  // COST_MODEL_COMMUNI_CONST
  double costmodel_communi_const_;

  // COST_MODEL_COMMUNI_BIAS
  double costmodel_communi_bias_;

  // MULTI_SUBGRAPHS
  bool is_multi_subgraphs_;

  // In the recovery phase of DP algorithm, when encountering triangle structure and star structure,
  // whether overwrite the right-node strategy
  bool triangle_star_strategy_overwrite_;

  // Whether to enable APPROXIMATION in the DP algorithm.
  bool dp_algo_enable_approxi_;

  // When APPROXIMATION is enabled in the DP algorithm, the 'epsilon' value used in the APPROXIMATION.
  double dp_algo_approxi_epsilon_;

  // In RP algorithm, the coef value can adjust the balance between data parallel and model parallel.
  double rp_matmul_mem_coef_;

  // Whether to generate a single suite of OperatorInfo for a loop.
  bool dp_algo_single_loop_;

  int64_t run_phase_;  // 0: 'training', 1: 'inference'

  int64_t costmodel_allreduce_fusion_algorithm_;

  int64_t costmodel_allreduce_fusion_times_;

  double costmodel_allreduce_fusion_tail_percent_;

  double costmodel_allreduce_fusion_tail_time_;

  double costmodel_allreduce_fusion_allreduce_inherent_time_;

  double costmodel_allreduce_fusion_allreduce_bandwidth_;

  double costmodel_allreduce_fusion_computation_time_parameter_;

  // TENSOR_SLICE_ALIGNMENT_ENABLE
  bool tensor_slice_alignment_enable_;

  // TENSOR_SLICE_ALIGNMENT_SIZE
  size_t tensor_slice_alignment_size_;

  // FULLY_USE_DEVICES
  bool fully_use_device_;

  // ELEMENTWISE_OP_STRA_FOLLOW
  bool elementwise_stra_follow_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_COSTMODEL_CONTEXT_H_
