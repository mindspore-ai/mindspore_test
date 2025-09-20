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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
class ValueTuple;
using ValueTuplePtr = std::shared_ptr<ValueTuple>;

namespace parallel {
constexpr size_t SORT_EXT_INPUT_SIZE = 2;

class ActivationBase : public OperatorInfo {
 public:
  ActivationBase(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ActivationBase() override = default;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferOutputTensorMap() override;
  Status InferDevMatrixShape() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputTensorInfo() override;
  virtual Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode);
  void set_output_infer_tensor_layout(const TensorLayout &tensor_layout) {
    output_infer_tensor_layout_ = tensor_layout;
  }
  size_t outputs_size_ = 1;

 private:
  TensorLayout output_infer_tensor_layout_;
};

class Activation : public ActivationBase {
 public:
  Activation(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~Activation() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
};

class ActivationInfo : public Activation {
 public:
  ActivationInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : Activation(name, inputs_shape, outputs_shape, attrs, std::make_shared<ActivationInfoCost>()) {}
  ~ActivationInfo() override = default;

 protected:
  Status GetAttrs() override;  // activation_type: relu, relu6, sigmoid
};

class ActivationOther : public Activation {
 public:
  ActivationOther(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : Activation(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ActivationOther() override = default;

 protected:
  Status GetAttrs() override;
};

class GeLUInfo : public ActivationOther {
 public:
  GeLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<GeLUCost>()) {}
  ~GeLUInfo() override = default;

 protected:
  Status InferForwardCommunicationByLayout() override;
};

class ClampScalarInfo : public GeLUInfo {
 public:
  ClampScalarInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : GeLUInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~ClampScalarInfo() override = default;
};

class FastGeLUInfo : public ActivationOther {
 public:
  FastGeLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<FastGeLUCost>()) {}
  ~FastGeLUInfo() override = default;
};

class TanhInfo : public ActivationOther {
 public:
  TanhInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<TanhCost>()) {}
  ~TanhInfo() override = default;
};

class Softmax : public ActivationBase {
 public:
  explicit Softmax(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftmaxCost>()) {}
  ~Softmax() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status CheckInputLayout() override;
  Status CheckLayoutConfig() override;
  Status GetAttrs() override;
  std::vector<int64_t> axis_;
  Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) override;
};

class SoftmaxInfo : public Softmax {
 public:
  SoftmaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~SoftmaxInfo() override = default;
};

class LogSoftmaxInfo : public Softmax {
 public:
  LogSoftmaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~LogSoftmaxInfo() override = default;

 protected:
  Status GetAttrs() override;
};

class SortInfo : public Softmax {
 public:
  SortInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~SortInfo() override = default;

 protected:
  Status InferTensorMap() override;
  Status InferAsLossDivisor() override;
  Status GetAttrs() override;
};

class SortExtInfo : public SortInfo {
 public:
  SortExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : SortInfo(name, inputs_shape, outputs_shape, attrs) {
    outputs_size_ = SORT_EXT_INPUT_SIZE;
  }
  ~SortExtInfo() override = default;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status InferAsLossDivisorByLayout() override;
  Status CheckInputLayout() override;
};

class ReverseV2Info : public Softmax {
 public:
  ReverseV2Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~ReverseV2Info() override = default;
};

class CumOpBase : public ActivationBase {
 public:
  CumOpBase(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~CumOpBase() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status GetAttrs() override;
  int64_t axis_ = -1;
  bool is_axis_ = true;
};

class CumSumInfo : public CumOpBase {
 public:
  CumSumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : CumOpBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<CumSumCost>()) {}
  ~CumSumInfo() override = default;
};

class CumsumExtInfo : public CumOpBase {
 public:
  CumsumExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : CumOpBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<CumsumExtCost>()) {
    is_axis_ = False;
  }
  ~CumsumExtInfo() override = default;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
};

class CumProdInfo : public CumOpBase {
 public:
  CumProdInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : CumOpBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<CumProdCost>()) {}
  ~CumProdInfo() = default;
};

class CummaxInfo : public CumOpBase {
 public:
  CummaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : CumOpBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<CumProdCost>()) {}
  ~CummaxInfo() = default;

 protected:
  Status InferMirrorOps() override;
  Status GetAttrs() override;        // the axis is in the attr
  Status InferTensorMap() override;  // cummax/cummin has two outputs
  Status InferAsLossDivisor() override;
};

class CumminInfo : public CummaxInfo {
 public:
  CumminInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : CummaxInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~CumminInfo() = default;
};

class EluInfo : public ActivationOther {
 public:
  EluInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<EluCost>()) {}
  ~EluInfo() override = default;
};

class EluExtInfo : public EluInfo {
 public:
  EluExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : EluInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~EluExtInfo() override = default;
};

class LeakyReLUExtInfo : public EluInfo {
 public:
  LeakyReLUExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : EluInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~LeakyReLUExtInfo() override = default;
};

class ReLUInfo : public ActivationOther {
 public:
  ReLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLUCost>()) {}
  ~ReLUInfo() override = default;
};

class SiLUInfo : public ActivationOther {
 public:
  SiLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SiLUCost>()) {}
  ~SiLUInfo() override = default;
};

class AShardIdentityInfo : public ReLUInfo {
 public:
  AShardIdentityInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : ReLUInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~AShardIdentityInfo() override = default;

 protected:
  Status CheckOutputLayout() override;
};

class IdentityInfo : public ActivationOther {
 public:
  IdentityInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<identityCost>()) {}
  ~IdentityInfo() override = default;
};

class RepeatElementsInfo : public ActivationOther {
 public:
  RepeatElementsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RepeatElementsCost>()) {}
  ~RepeatElementsInfo() override = default;
};

class ReLU6Info : public ActivationOther {
 public:
  ReLU6Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLU6Cost>()) {}
  ~ReLU6Info() override = default;
};

class SoftsignInfo : public ActivationOther {
 public:
  SoftsignInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftsignCost>()) {}
  ~SoftsignInfo() override = default;
};

class SoftplusInfo : public ActivationOther {
 public:
  SoftplusInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftplusCost>()) {}
  ~SoftplusInfo() override = default;
};

class SoftplusExtInfo : public SoftplusInfo {
 public:
  SoftplusExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : SoftplusInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~SoftplusExtInfo() override = default;
};

class CastInfo : public ActivationOther {
 public:
  CastInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<CastCost>()) {}
  ~CastInfo() override = default;

 protected:
  Status InferMirrorOps() override;
};

class SqrtInfo : public ActivationOther {
 public:
  SqrtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SqrtCost>()) {}
  ~SqrtInfo() override = default;
};

class NegInfo : public ActivationOther {
 public:
  NegInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<NegCost>()) {}
  ~NegInfo() override = default;
};

class ExpandDimsInfo : public ActivationOther {
 public:
  ExpandDimsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ExpandDimsCost>()) {}
  ~ExpandDimsInfo() override = default;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;

 private:
  int64_t positive_axis_ = -1;
  Strategies inputs_strategy_;
  Strategies outputs_strategy_;
};

class SqueezeInfo : public ActivationOther {
 public:
  SqueezeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SqueezeCost>()) {}
  ~SqueezeInfo() override = default;

 protected:
  Status InferAxis(const ValueTuplePtr &value_tuple);
  Status GetAttrs() override;
  void InferReplaceOps() override;
  Status InferTensorMap() override;

 private:
  ValueTuplePtr axis_;
};

class SquareInfo : public ActivationOther {
 public:
  SquareInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SquareCost>()) {}
  ~SquareInfo() override = default;
};

class SigmoidInfo : public ActivationOther {
 public:
  SigmoidInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SigmoidCost>()) {}
  ~SigmoidInfo() override = default;
};

class DropoutInfo : public ActivationOther {
 public:
  DropoutInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<DropOutCost>()) {}
  ~DropoutInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;
  void InferReplaceOps() override;
  Status InferAsLossDivisor() override;

 private:
  float keep_prob_ = 0.5;
  int64_t seed0_ = 0;
  int64_t seed1_ = 0;
  int64_t get_seed() const {
    static int64_t SEED_NUM = 0;
    return ++SEED_NUM;
  }
};

class DropoutExtInfo : public DropoutInfo {
 public:
  DropoutExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : DropoutInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~DropoutExtInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override;
  Status InferDevMatrixShape() override;
  void SetRepeatedCalcDevMatrix() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  void ReplaceNodeInputOrAttrs() override;

 private:
  static int64_t SEED_NUM;
  Shape mask_dev_matrix_shape_;  // dev matrix for `mask`
  bool IsUnsplittableStrategy(const Dimensions &strategy) const;
  CNodePtr GetGeneratorCNode(const CNodePtr &cnode) const;
  bool HaveManualSeed(const CNodePtr &generator_cnode) const;
  ParameterPtr GetSeedParameter(const CNodePtr &generator_cnode) const;
};

class HShrinkInfo : public ActivationOther {
 public:
  HShrinkInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<HShrinkCost>()) {}
  ~HShrinkInfo() = default;
};

class HSigmoidInfo : public ActivationOther {
 public:
  HSigmoidInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<HSigmoidCost>()) {}
  ~HSigmoidInfo() = default;
};

class IsFiniteInfo : public ActivationOther {
 public:
  IsFiniteInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<IsFiniteCost>()) {}
  ~IsFiniteInfo() = default;
};

class MishInfo : public ActivationOther {
 public:
  MishInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<MishCost>()) {}
  ~MishInfo() = default;
};

class RintInfo : public ActivationOther {
 public:
  RintInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RintCost>()) {}
  ~RintInfo() = default;
};

class SeLUInfo : public ActivationOther {
 public:
  SeLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SeLUCost>()) {}
  ~SeLUInfo() = default;
};

class SoftShrinkInfo : public ActivationOther {
 public:
  SoftShrinkInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftShrinkCost>()) {}
  ~SoftShrinkInfo() override = default;
};

class L2LossInfo : public ActivationOther {
 public:
  L2LossInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<L2LossCost>()) {}
  ~L2LossInfo() = default;

 protected:
  Status InferTensorMap() override;
  Status InferForwardCommunication() override;
};

class ErfinvInfo : public ActivationOther {
 public:
  ErfinvInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ErfinvCost>()) {}
  ~ErfinvInfo() override = default;
};

// the Invert has not backward
class InvertInfo : public ActivationOther {
 public:
  InvertInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLUCost>()) {}
  ~InvertInfo() = default;
};

// the PopulationCount has not backward
class PopulationCountInfo : public ActivationOther {
 public:
  PopulationCountInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLUCost>()) {}
  ~PopulationCountInfo() = default;
};

class NanToNumInfo : public ActivationOther {
 public:
  NanToNumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<NanToNumCost>()) {}
  ~NanToNumInfo() = default;
};

class RemainderTensorScalarInfo : public ActivationOther {
 public:
  RemainderTensorScalarInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RemainderCost>()) {}
  ~RemainderTensorScalarInfo() = default;
};

class RemainderScalarTensorInfo : public ActivationOther {
 public:
  RemainderScalarTensorInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RemainderCost>()) {}
  ~RemainderScalarTensorInfo() = default;

 protected:
  Status InferTensorInfo() override;
  Status InferMirrorOpsByLayout() override;
};

class SwigluInfo : public Softmax {
 public:
  SwigluInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~SwigluInfo() override = default;

 protected:
  Status GetAttrs() override;
  Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) override;
  Status InferOutputTensorInfo() override;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_
