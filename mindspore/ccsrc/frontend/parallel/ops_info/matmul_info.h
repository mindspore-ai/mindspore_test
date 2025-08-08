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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "utils/ms_utils.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t MATMUL_DIM = 2;
class MatMulBase : public OperatorInfo {
 public:
  // Generate all strategies and the corresponding cost for this MatMul operator
  MatMulBase(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~MatMulBase() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  Status SwapLastTwoElements(Shape *const input);

 protected:
  Status InferForwardCommunication() override;
  Status InferTensorInfo() override;  // the forward_reduce_scatter mode need to override this function
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout);
  Status GetAttrs() override;
  Status CheckBatchDimensions(const Dimensions &long_strategy, const Dimensions &short_strategy);
  Shape GetCommonShape(const Dimensions &mat_a_strategy, const Dimensions &mat_b_strategy) const;

  bool three_d_tp_ = false;
  bool enable_nd_tp_ = false;
  bool ndtp_split_y_ = false;
  int64_t ndtp_reduce_scatter_axis_ = -1;
  bool candidate_flag_ = false;
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  bool forward_reduce_scatter_ = false;
  int64_t field_size_ = 0;
  size_t mat_a_dimension_ = 0;
  size_t mat_b_dimension_ = 0;
  Shape origin_dev_matrix_shape_;
};

class MatMul : public MatMulBase {
 public:
  MatMul(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : MatMulBase(name, inputs_shape, outputs_shape, attrs) {}
  ~MatMul() override = default;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status CheckOutputStrategy(const StrategyPtr &out_strategy) override;
  Status InferOutputTensorMap() override;
  Status InferOutputTensorInfo() override;
  Status InferForwardCommunicationByLayout() override;
  Status CheckLayoutConfig() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  virtual Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode);
  StrategyPtr output_strategy_ = nullptr;

 private:
  void ProcessMatMulLeftInput(const std::vector<Group> &x_group_list, const AnfNodePtr &matmul_actual_input_node,
                              const size_t &all_gather_tensor_axis, const CNodePtr &cnode, GenerateGraph *gen_g,
                              AnfNodePtr *x_all_gather, AnfNodePtr *matmul_left_input);
  AnfNodePtr GetInputOutputNodeForNDTP(const CNodePtr &cnode, const AnfNodePtr &square_actual_input_node,
                                       GenerateGraph *gen_g, std::vector<std::pair<AnfNodePtr, int64_t>> *input_nodes);
  void CreateXZCommGroup(const TensorLayout &input_layout0, const int64_t &all_gather_dim0,
                         const int64_t &all_gather_dim1, std::vector<Group> *x_group_list,
                         std::vector<Group> *z_group_list);
  Status ComputeNDTPReplaceGraph(const CNodePtr &cnode);
  int64_t GetAllGatherDim(size_t all_gather_tensor_axis, const TensorLayout &input_layout, size_t tensor_index);
  AnfNodePtr ComputePreAllGatherGraph(const CNodePtr &cnode, GenerateGraph *gen_g,
                                      const std::vector<Group> &device_group_list, int64_t all_gather_tensor_axis,
                                      const AnfNodePtr &all_gather_node);
  AnfNodePtr ComputePostMatMulGraph(const CNodePtr &cnode, GenerateGraph *gen_g, const AnfNodePtr &matmul,
                                    const TensorLayout &input_layout, int64_t scatter_tensor_axis);
  TensorLayout InferNDTPOutputLayout();
  void Infer2DTPOutputLayout(const TensorLayout &input_layout0, const TensorLayout &input_layout1,
                             std::vector<Shape> *output_extended_tensor_map);
  void Infer3DTPOutputLayout(const TensorLayout &input_layout0, const TensorLayout &input_layout1,
                             std::vector<Shape> *output_extended_tensor_map);
  Status CheckNDTPInputLayout(const TensorLayout &a_in_layout, const TensorLayout &b_in_layout);
  Status Check2DTPInputLayout(const TensorLayout &a_in_layout, const TensorLayout &b_in_layout, size_t axis0_1,
                              size_t axis1_0, size_t axis1_1);
  Status Check3DTPInputLayout(const TensorLayout &a_in_layout, const TensorLayout &b_in_layout, size_t axis0_0,
                              size_t axis0_1, size_t axis1_0, size_t axis1_1);
  void CheckPCLMatMul(const Shape &mat_a_strategy, const Shape &mat_b_strategy);
  Status CheckInputStrategy(const Shape &mat_a_strategy, const Shape &mat_b_strategy);
  TensorLayout InferOutputLayout();
  TensorLayout output_infer_tensor_layout_;
};

class MatMulInfo : public MatMul {
 public:
  MatMulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : MatMul(name, inputs_shape, outputs_shape, attrs) {}
  ~MatMulInfo() override = default;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
};

class BatchMatMulInfo : public MatMul {
 public:
  BatchMatMulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : MatMul(name, inputs_shape, outputs_shape, attrs) {}
  ~BatchMatMulInfo() override = default;
  Status Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
              const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts = {},
              const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts = {}) override;
  Status CheckOutputStrategy(const StrategyPtr &out_strategy) override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
};

class MatMulExtInfo : public BatchMatMulInfo {
 public:
  MatMulExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : BatchMatMulInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~MatMulExtInfo() override = default;

 protected:
  Status GetAttrs() override;
};

class BatchMatMulExtInfo : public MatMulExtInfo {
 public:
  BatchMatMulExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : MatMulExtInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~BatchMatMulExtInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_
