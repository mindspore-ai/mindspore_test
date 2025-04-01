/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_OP_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_OP_H

#include <utility>
#include <tuple>
#include "plugin/device/ascend/kernel/dvm/lazy_fusion_kernel.h"
#include "kernel/ascend/pyboost/auto_generate/concat.h"
#include "kernel/ascend/pyboost/auto_generate/abs.h"
#include "kernel/ascend/pyboost/auto_generate/add.h"
#include "kernel/ascend/pyboost/auto_generate/cast.h"
#include "kernel/ascend/pyboost/auto_generate/mul.h"
#include "kernel/ascend/pyboost/auto_generate/muls.h"
#include "kernel/ascend/pyboost/auto_generate/sub.h"
#include "kernel/ascend/pyboost/auto_generate/exp.h"
#include "kernel/ascend/pyboost/auto_generate/div.h"
#include "kernel/ascend/pyboost/auto_generate/neg.h"
#include "kernel/ascend/pyboost/auto_generate/sqrt.h"
#include "kernel/ascend/pyboost/auto_generate/sigmoid.h"
#include "kernel/ascend/pyboost/auto_generate/sigmoid_grad.h"
#include "kernel/ascend/pyboost/auto_generate/silu.h"
#include "kernel/ascend/pyboost/auto_generate/silu_grad.h"
#include "kernel/ascend/pyboost/auto_generate/gelu.h"
#include "kernel/ascend/pyboost/auto_generate/gelu_grad.h"
#include "kernel/ascend/pyboost/auto_generate/relu.h"
#include "kernel/ascend/pyboost/auto_generate/sum_ext.h"
#include "kernel/ascend/pyboost/auto_generate/reciprocal.h"
#include "kernel/ascend/pyboost/auto_generate/isfinite.h"
#include "kernel/ascend/pyboost/auto_generate/round.h"
#include "kernel/ascend/pyboost/auto_generate/ceil.h"
#include "kernel/ascend/pyboost/auto_generate/floor.h"
#include "kernel/ascend/pyboost/auto_generate/trunc.h"
#include "kernel/ascend/pyboost/auto_generate/pow.h"
#include "kernel/ascend/pyboost/auto_generate/maximum.h"
#include "kernel/ascend/pyboost/auto_generate/minimum.h"
#include "kernel/ascend/pyboost/auto_generate/logical_not.h"
#include "kernel/ascend/pyboost/auto_generate/logical_and.h"
#include "kernel/ascend/pyboost/auto_generate/logical_or.h"
#include "kernel/ascend/pyboost/auto_generate/equal.h"
#include "kernel/ascend/pyboost/auto_generate/not_equal.h"
#include "kernel/ascend/pyboost/auto_generate/greater.h"
#include "kernel/ascend/pyboost/auto_generate/greater_equal.h"
#include "kernel/ascend/pyboost/auto_generate/less.h"
#include "kernel/ascend/pyboost/auto_generate/less_equal.h"
#include "kernel/ascend/pyboost/auto_generate/add_ext.h"
#include "kernel/ascend/pyboost/auto_generate/sub_ext.h"
#include "kernel/ascend/pyboost/auto_generate/tile.h"
#include "kernel/ascend/pyboost/auto_generate/linalg_vector_norm.h"
#include "kernel/ascend/pyboost/auto_generate/adamw.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_copy.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_div.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_exp.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_add_ext.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_sub_ext.h"
#include "kernel/ascend/pyboost/auto_generate/inplace_relu.h"
#include "kernel/ascend/pyboost/auto_generate/dense.h"
#include "kernel/ascend/pyboost/auto_generate/matmul.h"
#include "kernel/ascend/pyboost/auto_generate/batch_mat_mul.h"
#include "kernel/ascend/pyboost/auto_generate/matmul_ext.h"
#include "kernel/ascend/pyboost/auto_generate/batch_norm_stats.h"
#include "kernel/ascend/pyboost/auto_generate/batch_norm_gather_stats_with_counts.h"
#include "kernel/ascend/pyboost/auto_generate/batch_norm_elemt.h"
#include "kernel/ascend/pyboost/auto_generate/batch_norm_elemt_grad.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class ConcatAscendDvm : public ConcatAscend {
 public:
  ConcatAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : ConcatAscend(std::move(primitive), device_context) {}
  ~ConcatAscendDvm() = default;
  mindspore::tensor::BaseTensorPtr Call(const mindspore::ValueTuplePtr &tensors_tensor_list,
                                        const mindspore::Int64ImmPtr &axis) override;
};

class CastAscendDvm : public CastAscend {
 public:
  CastAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : CastAscend(std::move(primitive), device_context) {}
  ~CastAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const Int64ImmPtr &dtype) override;
};

class AbsAscendDvm : public AbsAscend {
 public:
  AbsAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : AbsAscend(std::move(primitive), device_context) {}
  ~AbsAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class NegAscendDvm : public NegAscend {
 public:
  NegAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : NegAscend(std::move(primitive), device_context) {}
  ~NegAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class ExpAscendDvm : public ExpAscend {
 public:
  ExpAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : ExpAscend(std::move(primitive), device_context) {}
  ~ExpAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class SqrtAscendDvm : public SqrtAscend {
 public:
  SqrtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SqrtAscend(std::move(primitive), device_context) {}
  ~SqrtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class ReciprocalAscendDvm : public ReciprocalAscend {
 public:
  ReciprocalAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : ReciprocalAscend(std::move(primitive), device_context) {}
  ~ReciprocalAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class IsFiniteAscendDvm : public IsFiniteAscend {
 public:
  IsFiniteAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : IsFiniteAscend(std::move(primitive), device_context) {}
  ~IsFiniteAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class RoundAscendDvm : public RoundAscend {
 public:
  RoundAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : RoundAscend(std::move(primitive), device_context) {}
  ~RoundAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const Int64ImmPtr &decimals) override;
};

class CeilAscendDvm : public CeilAscend {
 public:
  CeilAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : CeilAscend(std::move(primitive), device_context) {}
  ~CeilAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class FloorAscendDvm : public FloorAscend {
 public:
  FloorAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : FloorAscend(std::move(primitive), device_context) {}
  ~FloorAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class TruncAscendDvm : public TruncAscend {
 public:
  TruncAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : TruncAscend(std::move(primitive), device_context) {}
  ~TruncAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class EqualAscendDvm : public EqualAscend {
 public:
  EqualAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : EqualAscend(std::move(primitive), device_context) {}
  ~EqualAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class NotEqualAscendDvm : public NotEqualAscend {
 public:
  NotEqualAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : NotEqualAscend(std::move(primitive), device_context) {}
  ~NotEqualAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class GreaterAscendDvm : public GreaterAscend {
 public:
  GreaterAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : GreaterAscend(std::move(primitive), device_context) {}
  ~GreaterAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class GreaterEqualAscendDvm : public GreaterEqualAscend {
 public:
  GreaterEqualAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : GreaterEqualAscend(std::move(primitive), device_context) {}
  ~GreaterEqualAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class LessAscendDvm : public LessAscend {
 public:
  LessAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LessAscend(std::move(primitive), device_context) {}
  ~LessAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class LessEqualAscendDvm : public LessEqualAscend {
 public:
  LessEqualAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LessEqualAscend(std::move(primitive), device_context) {}
  ~LessEqualAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class AddAscendDvm : public AddAscend {
 public:
  AddAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : AddAscend(std::move(primitive), device_context) {}
  ~AddAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class MulAscendDvm : public MulAscend {
 public:
  MulAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MulAscend(std::move(primitive), device_context) {}
  ~MulAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class SubAscendDvm : public SubAscend {
 public:
  SubAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SubAscend(std::move(primitive), device_context) {}
  ~SubAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class DivAscendDvm : public DivAscend {
 public:
  DivAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : DivAscend(std::move(primitive), device_context) {}
  ~DivAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class PowAscendDvm : public PowAscend {
 public:
  PowAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : PowAscend(std::move(primitive), device_context) {}
  ~PowAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class MaximumAscendDvm : public MaximumAscend {
 public:
  MaximumAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MaximumAscend(std::move(primitive), device_context) {}
  ~MaximumAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class MinimumAscendDvm : public MinimumAscend {
 public:
  MinimumAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MinimumAscend(std::move(primitive), device_context) {}
  ~MinimumAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class LogicalNotAscendDvm : public LogicalNotAscend {
 public:
  LogicalNotAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LogicalNotAscend(std::move(primitive), device_context) {}
  ~LogicalNotAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor) override;
};

class LogicalAndAscendDvm : public LogicalAndAscend {
 public:
  LogicalAndAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LogicalAndAscend(std::move(primitive), device_context) {}
  ~LogicalAndAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class LogicalOrAscendDvm : public LogicalOrAscend {
 public:
  LogicalOrAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LogicalOrAscend(std::move(primitive), device_context) {}
  ~LogicalOrAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) override;
};

class MulsAscendDvm : public MulsAscend {
 public:
  MulsAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MulsAscend(std::move(primitive), device_context) {}
  ~MulsAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const ScalarPtr &other_tensor) override;
};

class SigmoidAscendDvm : public SigmoidAscend {
 public:
  SigmoidAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SigmoidAscend(std::move(primitive), device_context) {}
  ~SigmoidAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class SigmoidGradAscendDvm : public SigmoidGradAscend {
 public:
  SigmoidGradAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SigmoidGradAscend(std::move(primitive), device_context) {}
  ~SigmoidGradAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &y_tensor, const BaseTensorPtr &dy_tensor) override;
};

class SiLUAscendDvm : public SiLUAscend {
 public:
  SiLUAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SiLUAscend(std::move(primitive), device_context) {}
  ~SiLUAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class SiLUGradAscendDvm : public SiLUGradAscend {
 public:
  SiLUGradAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SiLUGradAscend(std::move(primitive), device_context) {}
  ~SiLUGradAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &dout_tensor, const BaseTensorPtr &x_tensor) override;
};

class GeLUAscendDvm : public GeLUAscend {
 public:
  GeLUAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : GeLUAscend(std::move(primitive), device_context) {}
  ~GeLUAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class GeLUGradAscendDvm : public GeLUGradAscend {
 public:
  GeLUGradAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : GeLUGradAscend(std::move(primitive), device_context) {}
  ~GeLUGradAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &dy_tensor, const BaseTensorPtr &x_tensor,
                             const BaseTensorPtr &y_tensor) override;
};

class ReLUAscendDvm : public ReLUAscend {
 public:
  ReLUAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : ReLUAscend(std::move(primitive), device_context) {}
  ~ReLUAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor) override;
};

class SumExtAscendDvm : public SumExtAscend {
 public:
  SumExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SumExtAscend(std::move(primitive), device_context) {}
  ~SumExtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const std::optional<ValueTuplePtr> &dim,
                             const BoolImmPtr &keepdim, const std::optional<Int64ImmPtr> &dtype) override;
};

class AddExtAscendDvm : public AddExtAscend {
 public:
  AddExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : AddExtAscend(std::move(primitive), device_context) {}
  ~AddExtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                             const ScalarPtr &alpha) override;
};

class SubExtAscendDvm : public SubExtAscend {
 public:
  SubExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : SubExtAscend(std::move(primitive), device_context) {}
  ~SubExtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                             const ScalarPtr &alpha) override;
};

class TileAscendDvm : public TileAscend {
 public:
  TileAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : TileAscend(std::move(primitive), device_context) {}
  ~TileAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const ValueTuplePtr &dims) override;
};

class LinalgVectorNormAscendDvm : public LinalgVectorNormAscend {
 public:
  LinalgVectorNormAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : LinalgVectorNormAscend(std::move(primitive), device_context) {}
  ~LinalgVectorNormAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor, const FP32ImmPtr &ord,
                             const std::optional<ValueTuplePtr> &dim, const BoolImmPtr &keepdim,
                             const std::optional<Int64ImmPtr> &dtype) override;
};

class AdamWAscendDvm : public AdamWAscend {
 public:
  AdamWAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : AdamWAscend(std::move(primitive), device_context) {}
  ~AdamWAscendDvm() = default;
  std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr> Call(
    const BaseTensorPtr &var_tensor, const BaseTensorPtr &m_tensor, const BaseTensorPtr &v_tensor,
    const BaseTensorPtr &max_v_tensor, const BaseTensorPtr &gradient_tensor, const BaseTensorPtr &step_tensor,
    const FP32ImmPtr &lr, const FP32ImmPtr &beta1, const FP32ImmPtr &beta2, const FP32ImmPtr &decay,
    const FP32ImmPtr &eps, const BoolImmPtr &amsgrad, const BoolImmPtr &maximize) override;
};

class InplaceCopyAscendDvm : public InplaceCopyAscend {
 public:
  InplaceCopyAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceCopyAscend(std::move(primitive), device_context) {}
  ~InplaceCopyAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &variable_tensor, const BaseTensorPtr &value_tensor) override;
};

class InplaceDivAscendDvm : public InplaceDivAscend {
 public:
  InplaceDivAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceDivAscend(std::move(primitive), device_context) {}
  ~InplaceDivAscendDvm() = default;
  tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &input_tensor,
                             const mindspore::tensor::BaseTensorPtr &other_tensor) override;
};

class InplaceExpAscendDvm : public InplaceExpAscend {
 public:
  InplaceExpAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceExpAscend(std::move(primitive), device_context) {}
  ~InplaceExpAscendDvm() = default;
  tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &input_tensor) override;
};

class InplaceAddExtAscendDvm : public InplaceAddExtAscend {
 public:
  InplaceAddExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceAddExtAscend(std::move(primitive), device_context) {}
  ~InplaceAddExtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                             const ScalarPtr &alpha) override;
};

class InplaceSubExtAscendDvm : public InplaceSubExtAscend {
 public:
  InplaceSubExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceSubExtAscend(std::move(primitive), device_context) {}
  ~InplaceSubExtAscendDvm() = default;
  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                             const ScalarPtr &alpha) override;
};

class InplaceReLUAscendDvm : public InplaceReLUAscend {
 public:
  InplaceReLUAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : InplaceReLUAscend(std::move(primitive), device_context) {}
  ~InplaceReLUAscendDvm() = default;
  tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &input_tensor) override;
};

class DenseAscendDvm : public DenseAscend {
 public:
  DenseAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : DenseAscend(std::move(primitive), device_context) {}
  ~DenseAscendDvm() = default;

  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &weight_tensor,
                             const std::optional<BaseTensorPtr> &bias_tensor) override;
};

class MatMulAscendDvm : public MatMulAscend {
 public:
  MatMulAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MatMulAscend(std::move(primitive), device_context) {}
  ~MatMulAscendDvm() = default;

  tensor::BaseTensorPtr Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &mat2_tensor,
                             const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) override;
};

class BatchMatMulAscendDvm : public pyboost::BatchMatMulAscend {
 public:
  BatchMatMulAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : BatchMatMulAscend(std::move(primitive), device_context) {}
  ~BatchMatMulAscendDvm() = default;

  tensor::BaseTensorPtr Call(const BaseTensorPtr &x_tensor, const BaseTensorPtr &y_tensor,
                             const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) override;
};

class MatMulExtAscendDvm : public MatMulExtAscend {
 public:
  MatMulExtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : MatMulExtAscend(std::move(primitive), device_context) {}
  ~MatMulExtAscendDvm() = default;

  tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &input_tensor,
                             const mindspore::tensor::BaseTensorPtr &other_tensor) override;
};

class BatchNormStatsAscendDvm : public BatchNormStatsAscend {
 public:
  BatchNormStatsAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : BatchNormStatsAscend(std::move(primitive), device_context) {}
  ~BatchNormStatsAscendDvm() = default;

  std::tuple<mindspore::tensor::BaseTensorPtr, mindspore::tensor::BaseTensorPtr> Call(
    const mindspore::tensor::BaseTensorPtr &input_tensor, const mindspore::FP32ImmPtr &eps) override;
};

class BatchNormGatherStatsWithCountsAscendDvm : public BatchNormGatherStatsWithCountsAscend {
 public:
  BatchNormGatherStatsWithCountsAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : BatchNormGatherStatsWithCountsAscend(std::move(primitive), device_context) {}
  ~BatchNormGatherStatsWithCountsAscendDvm() = default;

  std::tuple<mindspore::tensor::BaseTensorPtr, mindspore::tensor::BaseTensorPtr> Call(
    const mindspore::tensor::BaseTensorPtr &input_tensor, const mindspore::tensor::BaseTensorPtr &mean_tensor,
    const mindspore::tensor::BaseTensorPtr &invstd_tensor,
    const std::optional<mindspore::tensor::BaseTensorPtr> &running_mean_tensor_opt,
    const std::optional<mindspore::tensor::BaseTensorPtr> &running_var_tensor_opt,
    const mindspore::FP32ImmPtr &momentum, const mindspore::FP32ImmPtr &eps,
    const std::optional<mindspore::tensor::BaseTensorPtr> &counts_tensor_opt) override;
};

class BatchNormElemtAscendDvm : public BatchNormElemtAscend {
 public:
  BatchNormElemtAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : BatchNormElemtAscend(std::move(primitive), device_context) {}
  ~BatchNormElemtAscendDvm() = default;

  mindspore::tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &input_tensor,
                                        const std::optional<mindspore::tensor::BaseTensorPtr> &weight_tensor_opt,
                                        const std::optional<mindspore::tensor::BaseTensorPtr> &bias_tensor_opt,
                                        const std::optional<mindspore::tensor::BaseTensorPtr> &mean_tensor_opt,
                                        const std::optional<mindspore::tensor::BaseTensorPtr> &invstd_tensor_opt,
                                        const mindspore::FP32ImmPtr &eps) override;
};

class BatchNormElemtGradAscendDvm : public BatchNormElemtGradAscend {
 public:
  BatchNormElemtGradAscendDvm(PrimitivePtr primitive, const DeviceContext *device_context)
      : BatchNormElemtGradAscend(std::move(primitive), device_context) {}
  ~BatchNormElemtGradAscendDvm() = default;

  mindspore::tensor::BaseTensorPtr Call(const mindspore::tensor::BaseTensorPtr &dout_tensor,
                                        const mindspore::tensor::BaseTensorPtr &input_tensor,
                                        const mindspore::tensor::BaseTensorPtr &mean_tensor,
                                        const mindspore::tensor::BaseTensorPtr &invstd_tensor,
                                        const mindspore::tensor::BaseTensorPtr &weight_tensor,
                                        const mindspore::tensor::BaseTensorPtr &sumd_dy_tensor,
                                        const mindspore::tensor::BaseTensorPtr &sum_dy_xmu_tensor,
                                        const mindspore::tensor::BaseTensorPtr &count_tensor) override;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_OP_H
