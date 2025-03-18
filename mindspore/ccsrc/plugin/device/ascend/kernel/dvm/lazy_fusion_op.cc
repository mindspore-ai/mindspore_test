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

#include "plugin/device/ascend/kernel/dvm/lazy_fusion_op.h"
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include "base/bfloat16.h"
#include "infer/ops_func_impl/tile.h"
#include "plugin/device/ascend/kernel/dvm/lazy_fusion_kernel.h"
#include "plugin/device/ascend/kernel/dvm/lazy_fusion_flags.h"
#include "runtime/pipeline/pipeline.h"
#include "view/view_strides_calculator.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_n.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
bool IsFloatType(const TypeId &type) {
  switch (type) {
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeBFloat16:
      return true;
    default:
      return false;
  }
}

bool NeedSync() {
  static auto need_sync = runtime::OpExecutor::NeedSync();
  return (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph());
}

bool IsFloatIntType(const TypeId &type) { return IsFloatType(type) || type == kNumberTypeInt32; }

bool IsSupportType(const TypeId &type) { return IsFloatIntType(type) || type == kNumberTypeBool; }

bool InputCheck(const BaseTensorPtr &x, const std::function<bool(const TypeId &type)> &type_check = IsFloatType) {
  return !NeedSync() && x->is_contiguous() && type_check(x->data_type());
}

bool IsScalar(const BaseTensorPtr &x) { return (x->device_address() == nullptr) && (x->DataSize() == 1); }

template <typename T>
std::pair<bool, T> GetScalarValue(const ScalarPtr &s) {
  MS_EXCEPTION_IF_NULL(s);
  if (s->isa<Int64Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int64_t>(s)));
  } else if (s->isa<FP32Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<float>(s)));
  } else if (s->isa<BF16Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<bfloat16>(s)));
  } else if (s->isa<FP64Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<double>(s)));
  } else if (s->isa<Int32Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int32_t>(s)));
  } else if (s->isa<Int16Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int16_t>(s)));
  } else if (s->isa<Int8Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int8_t>(s)));
  } else if (s->isa<UInt8Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<uint8_t>(s)));
  }
  return std::make_pair(false, T(0));
}

ShapeVector GetReduceDim(const std::optional<ValueTuplePtr> &dim, size_t rank) {
  ShapeVector dim_value;
  dim_value.reserve(rank);
  if (dim.has_value()) {
    auto dim_data = dim.value()->value();
    (void)std::transform(dim_data.begin(), dim_data.end(), std::back_inserter(dim_value),
                         [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
      dim_value.push_back(i);
    }
  }
  return dim_value;
}

template <typename... Args>
void CheckForwardFuse(const device::DeviceContext *context, size_t stream, const Args &... inputs) {
  auto k = g_lazy_fusion_manager.Get(context, stream);
  bool fuse_forward = false;
  ((fuse_forward = fuse_forward || k->HasTensor(inputs)), ...);
  // flush if current op has no relation with previous ops
  if (!fuse_forward) {
    FlushLazyFusion();
  }
}

std::pair<bool, TypeId> CheckMatMul(const PrimitivePtr prim, const BaseTensorPtr &x_tensor,
                                    const BaseTensorPtr &y_tensor) {
  auto output_type = x_tensor->data_type();
  if (NeedSync()) {
    return {false, output_type};
  }
  if (output_type != y_tensor->data_type()) {
    return {false, output_type};
  }
  if (output_type != kNumberTypeFloat16 && output_type != kNumberTypeBFloat16) {
    return {false, output_type};
  }
  if (prim->HasAttr("cast_type")) {
    auto cast_type = prim->GetAttr("cast_type");
    if (!cast_type->isa<Type>() || !IsSupportType(output_type = cast_type->cast<TypePtr>()->type_id())) {
      return {false, output_type};
    }
  }
  if (!x_tensor->is_contiguous() || !y_tensor->is_contiguous()) {
    return {false, output_type};
  }
  if (x_tensor->shape().size() > kDim4 || y_tensor->shape().size() > kDim4) {
    return {false, output_type};
  }
  return {true, output_type};
}

template <typename F, typename... Args>
void DvmCall(const std::string &op_name, OpRunner *op, const F &func, const Args &... inputs) {
  op->ProfileTrackerTask();
  size_t stream = op->stream_id();
  const DeviceContext *context = op->device_context();
  auto k = g_lazy_fusion_manager.Get(context, stream);
  MS_LOG(INFO) << op_name << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(context, stream, inputs...);
  auto tensor = func(k);
  tensor->set_need_pipeline_sync(true);
  auto &outputs = const_cast<std::vector<tensor::BaseTensorPtr> &>(op->outputs());
  outputs.emplace_back(std::move(tensor));
  op->CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name << " call end, kernel id is " << k->id();
}

template <typename T>
T TensorToScalar(const tensor::BaseTensorPtr &tensor) {
  switch (tensor->data_type()) {
    case kNumberTypeBool:
      return static_cast<T>(static_cast<bool *>(tensor->data_c())[0]);
    case kNumberTypeFloat16:
      return static_cast<T>(static_cast<float16 *>(tensor->data_c())[0]);
    case kNumberTypeFloat32:
      return static_cast<T>(static_cast<float *>(tensor->data_c())[0]);
    case kNumberTypeInt32:
      return static_cast<T>(static_cast<int32_t *>(tensor->data_c())[0]);
    case kNumberTypeBFloat16:
      return static_cast<T>(static_cast<bfloat16 *>(tensor->data_c())[0]);
    default:
      return static_cast<T>(0);
  }
  return static_cast<T>(0);
}

void BinaryDvmCall(const std::string &op_name, OpRunner *op, dvm::BinaryOpType op_type,
                   const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor, const TypeId dst_type) {
  op->ProfileTrackerTask();
  size_t stream = op->stream_id();
  const DeviceContext *context = op->device_context();
  auto k = g_lazy_fusion_manager.Get(context, stream);
  MS_LOG(INFO) << op_name << " call start, kernel id is " << k->id();
  auto type_id = input_tensor->data_type();
  dvm::NDObject *obj = nullptr;
  if (IsScalar(input_tensor)) {
    PyBoostUtils::PrepareOpInputs(context, stream, other_tensor);
    if (type_id == kNumberTypeInt32) {
      obj = k->Binary(op_type, TensorToScalar<int32_t>(input_tensor), k->Input(other_tensor));
    } else {
      obj = k->Binary(op_type, TensorToScalar<float>(input_tensor), k->Input(other_tensor));
    }
  } else if (IsScalar(other_tensor)) {
    PyBoostUtils::PrepareOpInputs(context, stream, input_tensor);
    if (type_id == kNumberTypeInt32) {
      obj = k->Binary(op_type, k->Input(input_tensor), TensorToScalar<int32_t>(other_tensor));
    } else {
      obj = k->Binary(op_type, k->Input(input_tensor), TensorToScalar<float>(other_tensor));
    }
  } else {
    PyBoostUtils::PrepareOpInputs(context, stream, input_tensor, other_tensor);
    obj = k->Binary(op_type, k->Input(input_tensor), k->Input(other_tensor));
  }
  auto tensor = k->Output(obj, dst_type, k->GetShape(obj));

  tensor->set_need_pipeline_sync(true);
  auto &outputs = const_cast<std::vector<tensor::BaseTensorPtr> &>(op->outputs());
  outputs.emplace_back(std::move(tensor));
  op->CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name << " call end, kernel id is " << k->id();
}
}  // namespace

tensor::BaseTensorPtr CastAscendDvm::Call(const BaseTensorPtr &input_tensor, const Int64ImmPtr &dtype) {
  auto dst_type = static_cast<TypeId>(GetValue<int64_t>(dtype));
  if (!InputCheck(input_tensor, IsSupportType) || !IsSupportType(dst_type)) {
    return CastAscend::Call(input_tensor, dtype);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor, dst_type](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto src_obj = k->Input(input_tensor, false);
      auto dst_dtype = k->TransType(dst_type);
      auto obj = k->Cast(src_obj, dst_dtype);
      return k->Output(obj, dst_type, input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr AbsAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return AbsAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kAbs, k->Input(input_tensor));
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr NegAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return NegAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = input_tensor->data_type() == kNumberTypeInt32
                   ? k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), -1)
                   : k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), -1.0f);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr ExpAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return ExpAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kExp, k->Input(input_tensor));
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SqrtAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return SqrtAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kSqrt, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr ReciprocalAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return ReciprocalAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kReciprocal, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr IsFiniteAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return IsFiniteAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kIsFinite, k->Input(x_tensor));
      return k->Output(obj, kNumberTypeBool, x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr RoundAscendDvm::Call(const BaseTensorPtr &x_tensor, const Int64ImmPtr &decimals) {
  if (!InputCheck(x_tensor) || decimals->value() != 0) {
    return RoundAscend::Call(x_tensor, decimals);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kRound, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr CeilAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return CeilAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kCeil, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr FloorAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return FloorAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kFloor, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr TruncAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return TruncAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kTrunc, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr EqualAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return EqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kEqual, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr NotEqualAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return NotEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kNotEqual, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr GreaterAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return GreaterAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kGreater, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr GreaterEqualAscendDvm::Call(const BaseTensorPtr &input_tensor,
                                                  const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return GreaterEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kGreaterEqual, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr LessAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return LessAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kLess, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr LessEqualAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return LessEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kLessEqual, input_tensor, other_tensor, kNumberTypeBool);
  return outputs_.front();
}

tensor::BaseTensorPtr AddAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType) || !InputCheck(other_tensor, IsFloatIntType)) {
    return AddAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kAdd, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr MulAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType) || !InputCheck(other_tensor, IsFloatIntType)) {
    return MulAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMul, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr SubAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType) || !InputCheck(other_tensor, IsFloatIntType)) {
    return SubAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kSub, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr DivAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return DivAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kDiv, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr PowAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return PowAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kPow, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr MaximumAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return MaximumAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMaximum, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr MinimumAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType) || !InputCheck(other_tensor, IsFloatIntType)) {
    return MinimumAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMinimum, input_tensor, other_tensor, input_tensor->data_type());
  return outputs_.front();
}

tensor::BaseTensorPtr MulsAscendDvm::Call(const BaseTensorPtr &input_tensor, const ScalarPtr &other_tensor) {
  auto [succ, scalar] = GetScalarValue<float>(other_tensor);
  if (!InputCheck(input_tensor) || !succ) {
    return MulsAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto obj = k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), scalar);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr LogicalNotAscendDvm::Call(const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor, IsSupportType)) {
    return LogicalNotAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Cast(k->Input(x_tensor), dvm::DType::kBool);
      auto obj = k->Unary(dvm::UnaryOpType::kLogicalNot, input_obj);
      return k->Output(obj, kNumberTypeBool, x_tensor->shape());
    },
    x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr LogicalAndAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsSupportType) || !InputCheck(other_tensor, IsSupportType)) {
    return LogicalAndAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Cast(k->Input(input_tensor), dvm::DType::kBool);
      auto other_obj = k->Cast(k->Input(other_tensor), dvm::DType::kBool);
      auto obj = k->Binary(dvm::BinaryOpType::kLogicalAnd, input_obj, other_obj);
      return k->Output(obj, kNumberTypeBool, k->GetShape(obj));
    },
    input_tensor, other_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr LogicalOrAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor, IsSupportType) || !InputCheck(other_tensor, IsSupportType)) {
    return LogicalOrAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Cast(k->Input(input_tensor), dvm::DType::kBool);
      auto other_obj = k->Cast(k->Input(other_tensor), dvm::DType::kBool);
      auto obj = k->Binary(dvm::BinaryOpType::kLogicalOr, input_obj, other_obj);
      return k->Output(obj, kNumberTypeBool, k->GetShape(obj));
    },
    input_tensor, other_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SigmoidAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return SigmoidAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(input_tensor);
      auto input_type = k->GetDType(input_obj);
      auto need_cast = input_tensor->data_type() == kNumberTypeFloat16;
      if (need_cast) {
        input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
      }
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, input_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto obj = k->Unary(dvm::UnaryOpType::kReciprocal, add_exp);
      if (need_cast) {
        obj = k->Cast(obj, input_type);
      }
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SigmoidGradAscendDvm::Call(const BaseTensorPtr &y_tensor, const BaseTensorPtr &dy_tensor) {
  if (!InputCheck(y_tensor) || !InputCheck(dy_tensor)) {
    return SigmoidGradAscend::Call(y_tensor, dy_tensor);
  }
  DvmCall(
    op_name_, this,
    [&y_tensor, &dy_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto y_obj = k->Input(y_tensor);
      auto dy_obj = k->Input(dy_tensor);
      auto y_type = k->GetDType(y_obj);
      auto need_cast = y_tensor->data_type() == kNumberTypeFloat16;
      if (need_cast) {
        y_obj = k->Cast(y_obj, dvm::DType::kFloat32);
        dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
      }
      auto one_sub_y = k->Binary(dvm::BinaryOpType::kSub, 1.0f, y_obj);
      auto y_mul_dy = k->Binary(dvm::BinaryOpType::kMul, y_obj, dy_obj);
      auto obj = k->Binary(dvm::BinaryOpType::kMul, one_sub_y, y_mul_dy);
      if (need_cast) {
        obj = k->Cast(obj, y_type);
      }
      return k->Output(obj, y_tensor->data_type(), y_tensor->shape());
    },
    y_tensor, dy_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SiLUAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return SiLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(input_tensor);
      auto input_type = k->GetDType(input_obj);
      auto need_cast = input_tensor->data_type() == kNumberTypeFloat16;
      if (need_cast) {
        input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
      }
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, input_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto obj = k->Binary(dvm::BinaryOpType::kDiv, input_obj, add_exp);
      if (need_cast) {
        obj = k->Cast(obj, input_type);
      }
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SiLUGradAscendDvm::Call(const BaseTensorPtr &dout_tensor, const BaseTensorPtr &x_tensor) {
  if (!InputCheck(x_tensor) || !InputCheck(dout_tensor)) {
    return SiLUGradAscend::Call(dout_tensor, x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&dout_tensor, &x_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto dout_obj = k->Input(dout_tensor);
      auto x_obj = k->Input(x_tensor);
      auto x_type = k->GetDType(x_obj);
      auto need_cast = x_tensor->data_type() == kNumberTypeFloat16;
      if (need_cast) {
        x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
        dout_obj = k->Cast(dout_obj, dvm::DType::kFloat32);
      }
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, x_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto sigmod = k->Unary(dvm::UnaryOpType::kReciprocal, add_exp);
      auto out = k->Binary(dvm::BinaryOpType::kDiv, x_obj, add_exp);
      auto sigmod_out0 = k->Binary(dvm::BinaryOpType::kAdd, sigmod, out);
      auto sigmod_out1 = k->Binary(dvm::BinaryOpType::kMul, sigmod, out);
      auto sub_res = k->Binary(dvm::BinaryOpType::kSub, sigmod_out0, sigmod_out1);
      auto obj = k->Binary(dvm::BinaryOpType::kMul, sub_res, dout_obj);
      if (need_cast) {
        obj = k->Cast(obj, x_type);
      }
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    dout_tensor, x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr GeLUAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return GeLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto x_obj = k->Input(input_tensor);
      auto input_dtype = k->GetDType(x_obj);

      // Constants used in the GeLU approximation
      constexpr float csv_value = 0.044715f;
      constexpr float csv_value_sqrt_eight_div_pi = -1.5957691216057308f;
      x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
      // Compute x^2
      auto x_squared = k->Binary(dvm::BinaryOpType::kMul, x_obj, x_obj);
      // Compute x^3
      auto x_cubed = k->Binary(dvm::BinaryOpType::kMul, x_squared, x_obj);
      // mul_1 = x_cubed * csv_value
      auto mul_1 = k->Binary(dvm::BinaryOpType::kMul, x_cubed, csv_value);
      // tanh_res = x_obj + mul_1
      auto tanh_res = k->Binary(dvm::BinaryOpType::kAdd, x_obj, mul_1);
      // y = tanh_res * csv_value_sqrt_eight_div_pi
      auto y = k->Binary(dvm::BinaryOpType::kMul, tanh_res, csv_value_sqrt_eight_div_pi);
      // exp_0 = exp(y)
      auto exp_0 = k->Unary(dvm::UnaryOpType::kExp, y);
      // add_0 = exp_0 + 1
      auto add_0 = k->Binary(dvm::BinaryOpType::kAdd, exp_0, 1.0f);
      // result = x_obj / add_0
      auto result = k->Binary(dvm::BinaryOpType::kDiv, x_obj, add_0);
      result = k->Cast(result, input_dtype);
      return k->Output(result, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr GeLUGradAscendDvm::Call(const BaseTensorPtr &dy_tensor, const BaseTensorPtr &x_tensor,
                                              const BaseTensorPtr &y_tensor) {
  if (!InputCheck(dy_tensor) || !InputCheck(x_tensor)) {
    return GeLUGradAscend::Call(dy_tensor, x_tensor, y_tensor);
  }
  DvmCall(
    op_name_, this,
    [&dy_tensor, &x_tensor](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto dy_obj = k->Input(dy_tensor);
      auto x_obj = k->Input(x_tensor);
      auto input_dtype = k->GetDType(x_obj);

      // Constants used in the GeLU gradient computation
      constexpr float cs_value = 0.044715f;
      constexpr float cs_sqrt_two_div_pi = 0.7978845608028564f;  // sqrt(2 / π)
      constexpr float cs_value_tri = 0.134145f;                  // cs_value * 3
      x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
      dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
      // Compute x^2
      auto x_squared = k->Binary(dvm::BinaryOpType::kMul, x_obj, x_obj);
      // Compute x^3
      auto x_cubed = k->Binary(dvm::BinaryOpType::kMul, x_squared, x_obj);
      // Calculate mul_right
      auto mul_double_mul_tri = k->Binary(dvm::BinaryOpType::kMul, cs_value_tri, x_squared);
      auto mul_add_one = k->Binary(dvm::BinaryOpType::kAdd, 1.0f, mul_double_mul_tri);
      auto mul_right = k->Binary(dvm::BinaryOpType::kMul, cs_sqrt_two_div_pi, mul_add_one);
      // Calculate tanh_para
      auto mul_triple_mul_csvalue = k->Binary(dvm::BinaryOpType::kMul, cs_value, x_cubed);
      auto mul_add_x = k->Binary(dvm::BinaryOpType::kAdd, x_obj, mul_triple_mul_csvalue);
      auto tanh_para = k->Binary(dvm::BinaryOpType::kMul, cs_sqrt_two_div_pi, mul_add_x);
      // Compute tanh_res = tanh(tanh_para)
      auto Tanh = [&k](const auto &input) {
        // Implement tanh(x) = 1 - 2 / (e^{2x} + 1)
        auto two_input = k->Binary(dvm::BinaryOpType::kMul, input, 2.0f);
        auto exp_two_input = k->Unary(dvm::UnaryOpType::kExp, two_input);
        auto denom = k->Binary(dvm::BinaryOpType::kAdd, exp_two_input, 1.0f);
        auto two_div_denom = k->Binary(dvm::BinaryOpType::kDiv, 2.0f, denom);
        return k->Binary(dvm::BinaryOpType::kSub, 1.0f, two_div_denom);
      };
      auto tanh_res = Tanh(tanh_para);
      // Compute 0.5 * (1.0 + tanh_res)
      auto tanh_res_add_one = k->Binary(dvm::BinaryOpType::kAdd, 1.0f, tanh_res);
      auto half_mul_tanh_res_add_one = k->Binary(dvm::BinaryOpType::kMul, 0.5f, tanh_res_add_one);
      // Compute 1.0 - tanh_res^2
      auto tanh_res_squared = k->Binary(dvm::BinaryOpType::kMul, tanh_res, tanh_res);
      auto one_sub_tanh_res_squared = k->Binary(dvm::BinaryOpType::kSub, 1.0f, tanh_res_squared);
      // Compute 0.5 * x_obj * (1 - tanh_res^2)
      auto half_mul_x = k->Binary(dvm::BinaryOpType::kMul, 0.5f, x_obj);
      auto mul_tmp = k->Binary(dvm::BinaryOpType::kMul, half_mul_x, one_sub_tanh_res_squared);
      // Compute mul_final = mul_tmp * mul_right
      auto mul_final = k->Binary(dvm::BinaryOpType::kMul, mul_tmp, mul_right);
      // Compute result_tmp = half_mul_tanh_res_add_one + mul_final
      auto result_tmp = k->Binary(dvm::BinaryOpType::kAdd, half_mul_tanh_res_add_one, mul_final);
      // Compute result = dy_obj * result_tmp
      auto result = k->Binary(dvm::BinaryOpType::kMul, dy_obj, result_tmp);
      result = k->Cast(result, input_dtype);
      return k->Output(result, x_tensor->data_type(), x_tensor->shape());
    },
    dy_tensor, x_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SumExtAscendDvm::Call(const BaseTensorPtr &input_tensor, const std::optional<ValueTuplePtr> &dim,
                                            const BoolImmPtr &keepdim, const std::optional<Int64ImmPtr> &dtype) {
  auto dst_type = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : kMetaTypeNone;
  if (!InputCheck(input_tensor) || (dst_type != kMetaTypeNone && !IsFloatType(dst_type))) {
    return SumExtAscend::Call(input_tensor, dim, keepdim, dtype);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Cast(k->Input(input_tensor), dvm::DType::kFloat32);
      auto dim_value = GetReduceDim(dim, input_tensor->shape().size());
      auto reduce_obj = k->Reduce(dvm::ReduceOpType::kSum, input_obj, k->GetShapeRef(dim_value), keepdim->value());
      auto output_type = input_tensor->data_type();
      auto dst_dtype = k->TransType(output_type);
      if (dst_type != kMetaTypeNone) {
        dst_dtype = k->TransType(dst_type);
        output_type = dst_type;
      }
      reduce_obj = k->Cast(reduce_obj, dst_dtype);
      return k->Output(reduce_obj, output_type, k->GetShape(reduce_obj));
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr AddExtAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                                            const ScalarPtr &alpha) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return AddExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto [succ, scalar] = GetScalarValue<float>(alpha);
      if (!succ) {
        return AddExtAscend::Call(input_tensor, other_tensor, alpha);
      }
      auto input_obj = k->Input(input_tensor);
      auto other_obj = k->Input(other_tensor);
      if (scalar != 1.0) {
        other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
      }
      auto out_obj = k->Binary(dvm::BinaryOpType::kAdd, input_obj, other_obj);
      return k->Output(out_obj, input_tensor->data_type(), k->GetShape(out_obj));
    },
    input_tensor, other_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr SubExtAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                                            const ScalarPtr &alpha) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return SubExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto [succ, scalar] = GetScalarValue<float>(alpha);
      if (!succ) {
        return SubExtAscend::Call(input_tensor, other_tensor, alpha);
      }
      auto input_obj = k->Input(input_tensor);
      auto other_obj = k->Input(other_tensor);
      if (scalar != 1.0) {
        other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
      }
      auto out_obj = k->Binary(dvm::BinaryOpType::kSub, input_obj, other_obj);
      return k->Output(out_obj, input_tensor->data_type(), k->GetShape(out_obj));
    },
    input_tensor, other_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr TileAscendDvm::Call(const BaseTensorPtr &input_tensor, const ValueTuplePtr &dims) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return TileAscend::Call(input_tensor, dims);
  }
  auto input_shape = input_tensor->shape();
  ShapeVector output_shape = ConvertValueTupleToVector<int64_t>(dims);
  ops::AdaptShapeAndMultipies(&input_shape, &output_shape);
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (output_shape[i] != 1 && input_shape[i] != 1) {
      return TileAscend::Call(input_tensor, dims);
    }
    output_shape[i] *= input_shape[i];
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto out_obj = k->Broadcast(k->Input(input_tensor), k->GetShapeRef(output_shape));
      return k->Output(out_obj, input_tensor->data_type(), output_shape);
    },
    input_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr LinalgVectorNormAscendDvm::Call(const BaseTensorPtr &x_tensor, const FP32ImmPtr &ord,
                                                      const std::optional<ValueTuplePtr> &dim,
                                                      const BoolImmPtr &keepdim,
                                                      const std::optional<Int64ImmPtr> &dtype) {
  auto input_type = x_tensor->data_type();
  auto output_type = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : x_tensor->data_type();
  if (!IsFloatType(input_type) || !IsFloatType(output_type)) {
    return LinalgVectorNormAscend::Call(x_tensor, ord, dim, keepdim, dtype);
  }
  // if current reduce not fuse with its input, flush here to avoid generating a huge dvm kernel(e.g. global norm)
  CheckForwardFuse(device_context_, stream_id_, x_tensor);
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto dim_value = GetReduceDim(dim, x_tensor->shape().size());
      auto input_obj = k->Cast(k->Input(x_tensor), dvm::DType::kFloat32);
      dvm::NDObject *out_obj = nullptr;
      auto ord_value = ord->value();
      // sum(|x|^ord)^(1/ord)
      if (ord_value == 1.0f) {
        auto x_abs = k->Unary(dvm::UnaryOpType::kAbs, input_obj);
        out_obj = k->Reduce(dvm::ReduceOpType::kSum, x_abs, k->GetShapeRef(dim_value), keepdim->value());
      } else if (ord_value == 2.0f) {
        auto x_square = k->Binary(dvm::BinaryOpType::kMul, input_obj, input_obj);
        auto x_sum = k->Reduce(dvm::ReduceOpType::kSum, x_square, k->GetShapeRef(dim_value), keepdim->value());
        out_obj = k->Unary(dvm::UnaryOpType::kSqrt, x_sum);
      } else {
        auto x_abs = k->Unary(dvm::UnaryOpType::kAbs, input_obj);
        auto x_pow = k->Binary(dvm::BinaryOpType::kPow, x_abs, ord_value);
        auto x_sum = k->Reduce(dvm::ReduceOpType::kSum, x_pow, k->GetShapeRef(dim_value), keepdim->value());
        out_obj = k->Binary(dvm::BinaryOpType::kPow, x_sum, 1.0f / ord_value);
      }
      if (input_type != kNumberTypeBFloat16) {
        out_obj = k->Cast(out_obj, k->TransType(input_type));
      }
      return k->Output(out_obj, output_type, k->GetShape(out_obj));
    },
    x_tensor);
  return outputs_.front();
}

std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr> AdamWAscendDvm::Call(
  const BaseTensorPtr &var_tensor, const BaseTensorPtr &m_tensor, const BaseTensorPtr &v_tensor,
  const BaseTensorPtr &max_v_tensor, const BaseTensorPtr &gradient_tensor, const BaseTensorPtr &step_tensor,
  const FP32ImmPtr &lr, const FP32ImmPtr &beta1, const FP32ImmPtr &beta2, const FP32ImmPtr &decay,
  const FP32ImmPtr &eps, const BoolImmPtr &amsgrad, const BoolImmPtr &maximize) {
  ProfileTrackerTask();
  auto var_type = var_tensor->data_type();
  bool all_contiguous = var_tensor->is_contiguous() && m_tensor->is_contiguous() && v_tensor->is_contiguous() &&
                        max_v_tensor->is_contiguous() && gradient_tensor->is_contiguous() &&
                        step_tensor->is_contiguous();
  const auto lr_imm = GetValue<float>(lr);
  const auto beta1_imm = GetValue<float>(beta1);
  const auto beta2_imm = GetValue<float>(beta2);
  const auto decay_imm = GetValue<float>(decay);
  const auto epsilon_imm = GetValue<float>(eps);
  const auto amsgrad_imm = GetValue<bool>(amsgrad);
  const auto maximize_imm = GetValue<bool>(maximize);
  // input check
  if (!all_contiguous || var_type != kNumberTypeFloat32 || m_tensor->data_type() != var_type ||
      v_tensor->data_type() != var_type || (maximize_imm && max_v_tensor->data_type() != var_type) ||
      gradient_tensor->data_type() != var_type) {
    return AdamWAscend::Call(var_tensor, m_tensor, v_tensor, max_v_tensor, gradient_tensor, step_tensor, lr, beta1,
                             beta2, decay, eps, amsgrad, maximize);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  if (amsgrad_imm) {
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, var_tensor, m_tensor, v_tensor, max_v_tensor,
                                  gradient_tensor, step_tensor);
  } else {
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, var_tensor, m_tensor, v_tensor, gradient_tensor,
                                  step_tensor);
  }
  auto var_obj = k->Input(var_tensor);
  auto m_obj = k->Input(m_tensor);
  auto v_obj = k->Input(v_tensor);
  auto gradient_obj = k->Input(gradient_tensor);
  auto step_obj = k->Input(step_tensor);
  auto grad = maximize_imm ? k->Binary(dvm::BinaryOpType::kMul, gradient_obj, -1.0f) : gradient_obj;
  // m_t <-- beta1 * m + (1 - beta1) * grad
  auto m_t = k->Binary(dvm::BinaryOpType::kAdd, k->Binary(dvm::BinaryOpType::kMul, m_obj, beta1_imm),
                       k->Binary(dvm::BinaryOpType::kMul, grad, 1.0f - beta1_imm));
  // v_t <-- beta2 * v + (1 - beta2) * grad^2
  auto v_t =
    k->Binary(dvm::BinaryOpType::kAdd, k->Binary(dvm::BinaryOpType::kMul, v_obj, beta2_imm),
              k->Binary(dvm::BinaryOpType::kMul, k->Binary(dvm::BinaryOpType::kMul, grad, grad), 1.0f - beta2_imm));
  // var_t <-- var - lr * deday * var
  auto var_t = decay_imm == 0.0f ? var_obj : k->Binary(dvm::BinaryOpType::kMul, var_obj, 1.0f - lr_imm * decay_imm);
  // real step is step + 1
  auto step_value = step_tensor->data_type() == var_type ? step_obj : k->Cast(step_obj, k->TransType(var_type));
  step_value = k->Binary(dvm::BinaryOpType::kAdd, step_value, 1.0f);
  // var_t <-- var_t - (m_t * lr / (1 - beta1^t)) / (sqrt(max(max_v, v_t) / (1 - beta2^t)) + eps)
  auto scale1 =
    k->Binary(dvm::BinaryOpType::kDiv, lr_imm,
              k->Binary(dvm::BinaryOpType::kSub, 1.0f, k->Binary(dvm::BinaryOpType::kPow, beta1_imm, step_value)));
  auto m_t_scale = k->Binary(dvm::BinaryOpType::kMul, m_t, scale1);
  auto v_t_used = amsgrad_imm ? k->Binary(dvm::BinaryOpType::kMaximum, k->Input(max_v_tensor), v_t) : v_t;
  auto scale2 =
    k->Binary(dvm::BinaryOpType::kDiv, 1.0f,
              k->Binary(dvm::BinaryOpType::kSub, 1.0f, k->Binary(dvm::BinaryOpType::kPow, beta2_imm, step_value)));
  auto v_t_sqrt = k->Unary(dvm::UnaryOpType::kSqrt, k->Binary(dvm::BinaryOpType::kMul, v_t_used, scale2));
  var_t =
    k->Binary(dvm::BinaryOpType::kSub, var_t,
              k->Binary(dvm::BinaryOpType::kDiv, m_t_scale, k->Binary(dvm::BinaryOpType::kAdd, v_t_sqrt, epsilon_imm)));
  // update
  outputs_.push_back(var_tensor);
  k->Output(outputs_[kIndex0], var_t);
  outputs_.push_back(m_tensor);
  k->Output(outputs_[kIndex1], m_t);
  outputs_.push_back(v_tensor);
  k->Output(outputs_[kIndex2], v_t);
  for (const auto &output : outputs_) {
    output->set_need_pipeline_sync(true);
  }
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return std::make_tuple(outputs_[kIndex0], outputs_[kIndex1], outputs_[kIndex2]);
}

tensor::BaseTensorPtr InplaceCopyAscendDvm::Call(const BaseTensorPtr &variable_tensor,
                                                 const BaseTensorPtr &value_tensor) {
  if (!InputCheck(variable_tensor, IsFloatIntType) || !InputCheck(value_tensor, IsFloatIntType)) {
    return InplaceCopyAscend::Call(variable_tensor, value_tensor);
  }
  ProfileTrackerTask();
  auto addr0 = variable_tensor->device_address();
  auto addr1 = value_tensor->device_address();
  if (addr0 && addr1 && addr0->GetMutablePtr() == addr1->GetMutablePtr() &&
      value_tensor->shape() == variable_tensor->shape()) {
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, variable_tensor, value_tensor);
    outputs_.push_back(variable_tensor);
    outputs_[0]->set_need_pipeline_sync(true);
    CreateOutputSimpleInfo();
    FlushLazyFusion();
    return outputs_[0];
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, variable_tensor, value_tensor);
  // copy value_tensor to variable_tensor
  auto value_obj = k->Input(value_tensor, false);
  if (value_tensor->shape() != variable_tensor->shape()) {
    value_obj = k->Broadcast(value_obj, k->GetShapeRef(variable_tensor->shape()));
  }
  outputs_.push_back(variable_tensor);
  k->Output(outputs_[0], value_obj);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return outputs_[0];
}

tensor::BaseTensorPtr InplaceDivAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return InplaceDivAscend::Call(input_tensor, other_tensor);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, input_tensor, other_tensor);
  auto input_obj = k->Input(input_tensor);
  auto other_obj = k->Input(other_tensor);
  auto input_dtype = k->GetDType(input_obj);
  auto other_dtype = k->GetDType(other_obj);
  if (other_dtype != input_dtype) {
    other_obj = k->Cast(other_obj, input_dtype);
  }
  auto out_obj = k->Binary(dvm::BinaryOpType::kDiv, input_obj, other_obj);
  // update
  outputs_.push_back(input_tensor);
  k->Output(outputs_[0], out_obj);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return outputs_[0];
}

tensor::BaseTensorPtr InplaceExpAscendDvm::Call(const BaseTensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return InplaceExpAscend::Call(input_tensor);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, input_tensor);
  auto out_obj = k->Unary(dvm::UnaryOpType::kExp, k->Input(input_tensor));
  // update
  outputs_.push_back(input_tensor);
  k->Output(outputs_[0], out_obj);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return outputs_[0];
}

tensor::BaseTensorPtr InplaceAddExtAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                                                   const ScalarPtr &alpha) {
  auto [succ, scalar] = GetScalarValue<float>(alpha);
  if (!succ || !InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return InplaceAddExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, input_tensor, other_tensor);
  auto input_obj = k->Input(input_tensor);
  auto other_obj = k->Input(other_tensor);
  auto input_dtype = k->GetDType(input_obj);
  auto other_dtype = k->GetDType(other_obj);
  if (other_dtype != input_dtype) {
    other_obj = k->Cast(other_obj, input_dtype);
  }
  if (scalar != 1.0) {
    other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
  }
  auto out_obj = k->Binary(dvm::BinaryOpType::kAdd, input_obj, other_obj);
  // update
  outputs_.push_back(input_tensor);
  k->Output(outputs_[0], out_obj);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return outputs_[0];
}

tensor::BaseTensorPtr InplaceSubExtAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                                                   const ScalarPtr &alpha) {
  auto [succ, scalar] = GetScalarValue<float>(alpha);
  if (!succ || !InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return InplaceSubExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  MS_LOG(INFO) << op_name() << " call start, kernel id is " << k->id();
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, input_tensor, other_tensor);
  auto input_obj = k->Input(input_tensor);
  auto other_obj = k->Input(other_tensor);
  auto input_dtype = k->GetDType(input_obj);
  auto other_dtype = k->GetDType(other_obj);
  if (other_dtype != input_dtype) {
    other_obj = k->Cast(other_obj, input_dtype);
  }
  if (scalar != 1.0) {
    other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
  }
  auto out_obj = k->Binary(dvm::BinaryOpType::kSub, input_obj, other_obj);
  // update
  outputs_.push_back(input_tensor);
  k->Output(outputs_[0], out_obj);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  MS_LOG(INFO) << op_name() << " call end, kernel id is " << k->id();
  FlushLazyFusion();
  return outputs_[0];
}

tensor::BaseTensorPtr DenseAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &weight_tensor,
                                           const std::optional<BaseTensorPtr> &bias_tensor) {
  BaseTensorPtr bias = nullptr;
  if (bias_tensor.has_value()) {
    bias = bias_tensor.value();
    if (bias->shape().size() != kDim1 || !bias->is_contiguous()) {
      return DenseAscend::Call(input_tensor, weight_tensor, bias_tensor);
    }
  }
  if (!CheckMatMul(primitive_, input_tensor, weight_tensor).first) {
    return DenseAscend::Call(input_tensor, weight_tensor, bias_tensor);
  }
  FlushLazyFusion();  // forward fusion not allowed
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(input_tensor, false);
      auto weight_obj = k->Input(weight_tensor, false);
      auto bias_obj = bias == nullptr ? nullptr : k->Input(bias, false);
      auto out_obj = k->MatMul(input_obj, weight_obj, false, true, bias_obj);
      return k->Output(out_obj, input_tensor->data_type(), k->GetShape(out_obj));
    },
    input_tensor, weight_tensor, bias_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr MatMulAscendDvm::Call(const BaseTensorPtr &input_tensor, const BaseTensorPtr &mat2_tensor,
                                            const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) {
  auto [enable, output_type] = CheckMatMul(primitive_, input_tensor, mat2_tensor);
  if (!enable) {
    return MatMulAscend::Call(input_tensor, mat2_tensor, transpose_a, transpose_b);
  }
  FlushLazyFusion();  // forward fusion not allowed
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(input_tensor, false);
      auto weight_obj = k->Input(mat2_tensor, false);
      auto trans_a = GetValue<bool>(transpose_a);
      auto trans_b = GetValue<bool>(transpose_b);
      auto out_obj = k->MatMul(input_obj, weight_obj, trans_a, trans_b, nullptr);
      out_obj = k->Cast(out_obj, k->TransType(output_type));
      return k->Output(out_obj, output_type, k->GetShape(out_obj));
    },
    input_tensor, mat2_tensor);
  return outputs_.front();
}

tensor::BaseTensorPtr BatchMatMulAscendDvm::Call(const BaseTensorPtr &x_tensor, const BaseTensorPtr &y_tensor,
                                                 const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) {
  auto [enable, output_type] = CheckMatMul(primitive_, x_tensor, y_tensor);
  if (!enable) {
    return BatchMatMulAscend::Call(x_tensor, y_tensor, transpose_a, transpose_b);
  }
  FlushLazyFusion();  // forward fusion not allowed
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(x_tensor, false);
      auto weight_obj = k->Input(y_tensor, false);
      auto trans_a = GetValue<bool>(transpose_a);
      auto trans_b = GetValue<bool>(transpose_b);
      auto out_obj = k->MatMul(input_obj, weight_obj, trans_a, trans_b, nullptr);
      out_obj = k->Cast(out_obj, k->TransType(output_type));
      return k->Output(out_obj, output_type, k->GetShape(out_obj));
    },
    x_tensor, y_tensor);
  return outputs_.front();
}

bool CheckMatMulExtTranspose(const mindspore::tensor::BaseTensorPtr &tensor, bool *transpose, ShapeVector *shape) {
  *transpose = false;
  const auto &tensor_shape = tensor->shape();
  *shape = tensor_shape;
  if (!tensor->is_contiguous()) {
    auto storage_info = tensor->storage_info();
    MS_EXCEPTION_IF_NULL(storage_info);
    const auto &cur_shape = storage_info->shape;
    const auto &cur_strides = storage_info->strides;
    const auto &ori_shape = storage_info->ori_shape;
    const auto &ori_strides = storage_info->ori_strides;
    if (ops::IsContiguous(ori_shape, ori_strides) && cur_strides.size() == kDim2 && cur_strides[0] == 1 &&
        cur_strides[1] == cur_shape[0]) {
      *transpose = true;
      (*shape).resize(kDim2);
      (*shape)[0] = cur_shape[1];
      (*shape)[1] = cur_shape[0];
      return true;
    }
    return false;
  }
  return true;
}

tensor::BaseTensorPtr MatMulExtAscendDvm::Call(const mindspore::tensor::BaseTensorPtr &input_tensor,
                                               const mindspore::tensor::BaseTensorPtr &other_tensor) {
  bool transpose_a = false;
  bool transpose_b = false;
  ShapeVector input_shape;
  ShapeVector other_shape;
  auto data_type = input_tensor->data_type();
  if (NeedSync() || other_tensor->data_type() != data_type ||
      (data_type != kNumberTypeFloat16 && data_type != kNumberTypeBFloat16) ||
      !CheckMatMulExtTranspose(input_tensor, &transpose_a, &input_shape) ||
      !CheckMatMulExtTranspose(other_tensor, &transpose_b, &other_shape)) {
    return MatMulExtAscend::Call(input_tensor, other_tensor);
  }
  FlushLazyFusion();  // forward fusion not allowed
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> BaseTensorPtr {
      auto input_obj = k->Input(input_tensor, false, input_shape);
      auto weight_obj = k->Input(other_tensor, false, other_shape);
      auto out_obj = k->MatMul(input_obj, weight_obj, transpose_a, transpose_b, nullptr);
      return k->Output(out_obj, data_type, k->GetShape(out_obj));
    },
    input_tensor, other_tensor);
  return outputs_.front();
}

#define MS_REPLACE_DVM_OP(clazz)                                                                     \
  if (std::find(disable_ops.begin(), disable_ops.end(), #clazz) == disable_ops.end()) {              \
    OpFactory<clazz>::Get().op_creator()[kAscendDevice] = []() {                                     \
      return std::make_shared<clazz##AscendDvm>(prim::kPrim##clazz,                                  \
                                                runtime::OpRunner::GetDeviceContext(kAscendDevice)); \
    };                                                                                               \
  }

void RegisterLazyFusionOp() {
  const auto &disable_ops = LazyFusionFlags::GetInstance().disable_ops;
  MS_REPLACE_DVM_OP(Add);
  MS_REPLACE_DVM_OP(Cast);
  MS_REPLACE_DVM_OP(Abs);
  MS_REPLACE_DVM_OP(Neg);
  MS_REPLACE_DVM_OP(Exp);
  MS_REPLACE_DVM_OP(Sqrt);
  MS_REPLACE_DVM_OP(Reciprocal);
  MS_REPLACE_DVM_OP(IsFinite);
  MS_REPLACE_DVM_OP(Round);
  MS_REPLACE_DVM_OP(Ceil);
  MS_REPLACE_DVM_OP(Floor);
  MS_REPLACE_DVM_OP(Trunc);
  MS_REPLACE_DVM_OP(Equal);
  MS_REPLACE_DVM_OP(NotEqual);
  MS_REPLACE_DVM_OP(Greater);
  MS_REPLACE_DVM_OP(GreaterEqual);
  MS_REPLACE_DVM_OP(Less);
  MS_REPLACE_DVM_OP(LessEqual);
  MS_REPLACE_DVM_OP(Add);
  MS_REPLACE_DVM_OP(Mul);
  MS_REPLACE_DVM_OP(Muls);
  MS_REPLACE_DVM_OP(Sub);
  MS_REPLACE_DVM_OP(Div);
  MS_REPLACE_DVM_OP(Pow);
  MS_REPLACE_DVM_OP(Maximum);
  MS_REPLACE_DVM_OP(Minimum);
  MS_REPLACE_DVM_OP(LogicalNot);
  MS_REPLACE_DVM_OP(LogicalAnd);
  MS_REPLACE_DVM_OP(LogicalOr);
  MS_REPLACE_DVM_OP(Sigmoid);
  MS_REPLACE_DVM_OP(SigmoidGrad);
  MS_REPLACE_DVM_OP(SiLU);
  MS_REPLACE_DVM_OP(SiLUGrad);
  MS_REPLACE_DVM_OP(GeLU);
  MS_REPLACE_DVM_OP(GeLUGrad);
  MS_REPLACE_DVM_OP(SumExt);
  MS_REPLACE_DVM_OP(AddExt);
  MS_REPLACE_DVM_OP(SubExt);
  MS_REPLACE_DVM_OP(Tile);
  MS_REPLACE_DVM_OP(LinalgVectorNorm);
  MS_REPLACE_DVM_OP(AdamW);
  MS_REPLACE_DVM_OP(InplaceCopy);
  MS_REPLACE_DVM_OP(InplaceDiv);
  MS_REPLACE_DVM_OP(InplaceExp);
  MS_REPLACE_DVM_OP(InplaceAddExt);
  MS_REPLACE_DVM_OP(InplaceSubExt);
  MS_REPLACE_DVM_OP(Dense);
  MS_REPLACE_DVM_OP(MatMul);
  MS_REPLACE_DVM_OP(BatchMatMul);
  MS_REPLACE_DVM_OP(MatMulExt);
}

void LazyFusionAscendInit() {
  if (LazyFusionFlags::GetInstance().opt_level < OptLevel_1 || runtime::RuntimeConf::GetInstance()->launch_blocking() ||
      MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON") {
    MS_LOG(INFO) << "Skip init lazy fusion.";
    return;
  }
  MS_LOG(INFO) << "Init lazy fusion.";
  RegisterLazyFusionOp();
  runtime::Pipeline::Get().UpdateBackendStage(
    std::make_unique<LazyFusionQueue>("backend_queue", runtime::kThreadWaitLevel::kLevelBackend));
  if (LazyFusionFlags::GetInstance().online_tuning) {
    dvm::SetOnlineTuning(true);
  }
}

MS_REGISTER_LAZY_FUSION_INIT(kAscendDevice, LazyFusionAscendInit);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
