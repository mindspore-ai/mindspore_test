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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/activation_info.h"

namespace mindspore {
namespace parallel {
class ArithmeticBase : public OperatorInfo {
 public:
  ArithmeticBase(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ArithmeticBase() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  void ReComputeBatchSplitFlagList() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputTensorInfo() override;
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status BaseCheckStrategy(const StrategyPtr &strategy);
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferOutputTensorMap() override;
  Status CheckLayoutConfig() override;
  Shapes InferExpandShape();
  virtual Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode);
  virtual TensorLayout InferOutputLayout();
  TensorLayout output_infer_tensor_layout_;
};

class ArithmeticScalarBase : public Activation {
 public:
  ArithmeticScalarBase(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : Activation(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ArithmeticScalarBase() override = default;

 protected:
  Status GetAttrs() override { return SUCCESS; }
};

class FmodScalarInfo : public ArithmeticScalarBase {
 public:
  FmodScalarInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<FmodScalarCost>()) {}
  ~FmodScalarInfo() override = default;
};

class MulsInfo : public ArithmeticScalarBase {
 public:
  MulsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~MulsInfo() override = default;
};

class SubInfo : public ArithmeticBase {
 public:
  SubInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<SubCost>()) {}
  ~SubInfo() override = default;
};

class SubExtInfo : public ArithmeticBase {
 public:
  SubExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<SubCost>()) {}
  ~SubExtInfo() override = default;
};

class AddInfo : public ArithmeticBase {
 public:
  AddInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorAddCost>()) {}
  ~AddInfo() override = default;

 protected:
  Status InferForwardCommunicationByLayout() override { return SUCCESS; }
};

class AddExtInfo : public ArithmeticBase {
 public:
  AddExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorAddCost>()) {}
  ~AddExtInfo() override = default;
};

class MulInfo : public ArithmeticBase {
 public:
  MulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~MulInfo() override = default;
};

class AddcmulExtInfo : public ArithmeticBase {
 public:
  AddcmulExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<AddcmulExtCost>()) {}
  ~AddcmulExtInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferOutputTensorMap() override;
  Status CheckLayoutConfig() override;
  Status CheckInputLayout() override;
  TensorLayout InferOutputLayout() override;

 private:
  size_t inputs_size_ = 0;
  Strategies expand_strategies_;
  Dimensions broadcast_strategy_;
};

class DivInfo : public ArithmeticBase {
 public:
  DivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<DivCost>()) {}
  ~DivInfo() override = default;
};

class ModInfo : public ArithmeticBase {
 public:
  ModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ModCost>()) {}
  ~ModInfo() override = default;
};

class DivModInfo : public ArithmeticBase {
 public:
  DivModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ModCost>()) {}
  ~DivModInfo() override = default;
};

class RealDivInfo : public ArithmeticBase {
 public:
  RealDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReadDivCost>()) {}
  ~RealDivInfo() override = default;
};

class FloorDivInfo : public ArithmeticBase {
 public:
  FloorDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<FloorDivCost>()) {}
  ~FloorDivInfo() override = default;
};

class FloorModInfo : public ArithmeticBase {
 public:
  FloorModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<FloorModCost>()) {}
  ~FloorModInfo() override = default;
};

class PowInfo : public ArithmeticBase {
 public:
  PowInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<PowCost>()) {}
  ~PowInfo() override = default;
};

class AssignSubInfo : public ArithmeticBase {
 public:
  AssignSubInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<AssignSubCost>()) {}
  ~AssignSubInfo() override = default;
};

class AssignInfo : public ArithmeticBase {
 public:
  AssignInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<AssignCost>()) {}
  ~AssignInfo() override = default;
};

class AssignAddInfo : public ArithmeticBase {
 public:
  AssignAddInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<AssignAddCost>()) {}
  ~AssignAddInfo() override = default;
};

// All dimensions can be split arbitrarily, but the split method of Logits should be the same as that of label.
class SigmoidCrossEntropyWithLogitsInfo : public ArithmeticBase {
 public:
  SigmoidCrossEntropyWithLogitsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                                    const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs,
                       std::make_shared<SigmoidCrossEntropyWithLogitsCost>()) {}
  ~SigmoidCrossEntropyWithLogitsInfo() override = default;
};

class Atan2Info : public ArithmeticBase {
 public:
  Atan2Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<Atan2Cost>()) {}
  ~Atan2Info() override = default;
};

class DivNoNanInfo : public ArithmeticBase {
 public:
  DivNoNanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<DivNoNanCost>()) {}
  ~DivNoNanInfo() override = default;
};

class LogicalAndInfo : public ArithmeticBase {
 public:
  LogicalAndInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<LogicalAndCost>()) {}
  ~LogicalAndInfo() override = default;
};

class LogicalOrInfo : public ArithmeticBase {
 public:
  LogicalOrInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<LogicalOrCost>()) {}
  ~LogicalOrInfo() override = default;
};

class BitwiseAndInfo : public ArithmeticBase {
 public:
  BitwiseAndInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<BitwiseAndCost>()) {}
  ~BitwiseAndInfo() override = default;
};

class BitwiseOrInfo : public ArithmeticBase {
 public:
  BitwiseOrInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<BitwiseOrCost>()) {}
  ~BitwiseOrInfo() override = default;
};

class BitwiseXorInfo : public ArithmeticBase {
 public:
  BitwiseXorInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<BitwiseXorCost>()) {}
  ~BitwiseXorInfo() override = default;
};

class MulNoNanInfo : public ArithmeticBase {
 public:
  MulNoNanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulNoNanCost>()) {}
  ~MulNoNanInfo() = default;
};

class TruncateDivInfo : public ArithmeticBase {
 public:
  TruncateDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<TruncateDivCost>()) {}
  ~TruncateDivInfo() = default;
};

class TruncateModInfo : public ArithmeticBase {
 public:
  TruncateModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<TruncateModCost>()) {}
  ~TruncateModInfo() = default;
};

class XdivyInfo : public ArithmeticBase {
 public:
  XdivyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<XdivyCost>()) {}
  ~XdivyInfo() = default;
};

class OuterInfo : public ArithmeticBase {
 public:
  OuterInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, input_shape, output_shape, attrs, std::make_shared<OuterCost>()) {}
  ~OuterInfo() = default;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  Status CheckInputLayout() override;
  Status InferOutputTensorInfo() override;
};

class HypotInfo : public XdivyInfo {
 public:
  HypotInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~HypotInfo() = default;
};

class IgammaInfo : public XdivyInfo {
 public:
  IgammaInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~IgammaInfo() = default;
};

class IgammacInfo : public XdivyInfo {
 public:
  IgammacInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~IgammacInfo() = default;
};

class LeftShiftInfo : public XdivyInfo {
 public:
  LeftShiftInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~LeftShiftInfo() = default;
};

class RightShiftInfo : public XdivyInfo {
 public:
  RightShiftInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~RightShiftInfo() = default;
};

class NextAfterInfo : public XdivyInfo {
 public:
  NextAfterInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~NextAfterInfo() = default;
};

class ZetaInfo : public XdivyInfo {
 public:
  ZetaInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~ZetaInfo() = default;
};

class GcdInfo : public XdivyInfo {
 public:
  GcdInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~GcdInfo() = default;
};

class XlogyInfo : public ArithmeticBase {
 public:
  XlogyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<XlogyCost>()) {}
  ~XlogyInfo() = default;
};

class LerpInfo : public ArithmeticBase {
 public:
  LerpInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<LerpCost>()) {}
  ~LerpInfo() = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;

 private:
  size_t inputs_size_ = 0;
};

class SquaredDifferenceInfo : public ArithmeticBase {
 public:
  SquaredDifferenceInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape,
                        const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, input_shape, output_shape, attrs, std::make_shared<SquaredDifferenceCost>()) {}
  ~SquaredDifferenceInfo() override = default;
};

class MaskedFillInfo : public ArithmeticBase {
 public:
  MaskedFillInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &output_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, output_shape, attrs, std::make_shared<MaskedFillCost>()) {}
  ~MaskedFillInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;

 private:
  size_t input_size_ = 0;
};

class FmodTensorInfo : public XdivyInfo {
 public:
  FmodTensorInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~FmodTensorInfo() override = default;
};

class InplaceCopyInfo : public XdivyInfo {
 public:
  InplaceCopyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~InplaceCopyInfo() override = default;
};

class PolarInfo : public XdivyInfo {
 public:
  PolarInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~PolarInfo() override = default;
};

class IsCloseInfo : public XdivyInfo {
 public:
  IsCloseInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~IsCloseInfo() override = default;
};

class RemainderTensorTensorInfo : public XdivyInfo {
 public:
  RemainderTensorTensorInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                            const PrimitiveAttrs &attrs)
      : XdivyInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~RemainderTensorTensorInfo() override = default;
};

class InplaceAddExtInfo : public ArithmeticBase {
 public:
  InplaceAddExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~InplaceAddExtInfo() override = default;
};

class InplaceSubExtInfo : public ArithmeticBase {
 public:
  InplaceSubExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~InplaceSubExtInfo() override = default;
};

class InplaceMulInfo : public ArithmeticBase {
 public:
  InplaceMulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~InplaceMulInfo() override = default;
};

class InplaceDivInfo : public ArithmeticBase {
 public:
  InplaceDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~InplaceDivInfo() override = default;
};

class InplaceFloorDivideInfo : public ArithmeticBase {
 public:
  InplaceFloorDivideInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                         const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulCost>()) {}
  ~InplaceFloorDivideInfo() override = default;
};

class InplaceAddsExtInfo : public ArithmeticScalarBase {
 public:
  InplaceAddsExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~InplaceAddsExtInfo() override = default;
};

class InplaceSubScalarInfo : public ArithmeticScalarBase {
 public:
  InplaceSubScalarInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~InplaceSubScalarInfo() override = default;
};

class InplaceMulsInfo : public ArithmeticScalarBase {
 public:
  InplaceMulsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~InplaceMulsInfo() override = default;
};

class InplaceDivsInfo : public ArithmeticScalarBase {
 public:
  InplaceDivsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~InplaceDivsInfo() override = default;
};

class InplaceFloorDividesInfo : public ArithmeticScalarBase {
 public:
  InplaceFloorDividesInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : ArithmeticScalarBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MulsCost>()) {}
  ~InplaceFloorDividesInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_
