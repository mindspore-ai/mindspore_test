/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "mindspore/ops/infer/symbol_ops_impl/operator_scope.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
namespace {
constexpr size_t kNum2 = 2;
constexpr size_t kNum3 = 3;
constexpr size_t kNum4 = 4;
constexpr size_t kNum6 = 6;
}  // namespace
class OPS_API Conv : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Conv(const SymbolPtr &x, const SymbolPtr &out_channel, const SymbolPtr &kernel_size, const SymbolPtr &pad_mode,
       const SymbolPtr &padding, const SymbolPtr &stride, const SymbolPtr &dilation, const SymbolPtr &format)
      : InferShapeOp({x, out_channel, kernel_size, pad_mode, padding, stride, dilation, format}) {}

  ~Conv() override = default;
  MS_DECLARE_PARENT(Conv, InferShapeOp)

 protected:
  SymbolPtr CalcForPadValid(const SymbolPtr &x, const SymbolPtr &kernel, const SymbolPtr &stride,
                            const SymbolPtr &dilation);
  SymbolPtr CalcForPadSame(const SymbolPtr &x, const SymbolPtr &stride) {
    return Emit(std::make_shared<ScalarCeilDiv>(x, stride));
  }
  SymbolPtr CalcForPadding(const SymbolPtr &x_shape, const SymbolPtr &kernel, const SymbolPtr &padding,
                           const SymbolPtr &stride, const SymbolPtr &dilation);

  ListSymbolPtr ProcessAttr(const std::string &name, const SymbolPtr &attr, size_t begin_idx, size_t num) {
    if (attr->is<ListSymbol>()) {
      auto list = attr->as_sptr<ListSymbol>();
      if (list->size() < begin_idx + num) {
        MS_LOG(EXCEPTION) << "For " << name << ", attribute size is " << list->size() << ", begin index is "
                          << begin_idx << ", element number is " << num << ". attr: " << attr->ToString();
      }
      SymbolPtrList res(list->symbols().begin() + begin_idx, list->symbols().begin() + begin_idx + num);
      return ListSymbol::Make(std::move(res));
    }
    SymbolPtrList res(num, attr);
    return ListSymbol::Make(std::move(res));
  }

  int64_t ProcessPadMode(const SymbolPtr &pad_mode_sym) {
    int64_t pad_mode;
    if (pad_mode_sym->is<StrSymbol>()) {
      CheckAndConvertUtils::GetPadModEnumValue(MakeValue(pad_mode_sym->as<StrSymbol>()->value()), &pad_mode);
    } else if (pad_mode_sym->is<IntSymbol>()) {
      pad_mode = pad_mode_sym->as<IntSymbol>()->value();
    } else {
      MS_LOG(EXCEPTION) << "Unsupported pad_mode " << pad_mode_sym->ToString();
    }
    return pad_mode;
  }
};

SymbolPtr Conv::CalcForPadValid(const SymbolPtr &x, const SymbolPtr &kernel, const SymbolPtr &stride,
                                const SymbolPtr &dilation) {
  // `(x - (kernel - 1) * dilation)) / stride`, to ceil
  OperatorScope h(emitter(), OperatorScope::DivType::CEIL_DIV);
  auto v1 = h(kSym1);
  return (x - (kernel - v1) * dilation) / stride;
}

SymbolPtr Conv::CalcForPadding(const SymbolPtr &x_shape, const SymbolPtr &kernel, const SymbolPtr &padding,
                               const SymbolPtr &stride, const SymbolPtr &dilation) {
  // `[(x + padding - (kernel - 1) * dilation - 1) / stride] + 1`, [] is to floor.
  OperatorScope h(emitter(), OperatorScope::DivType::FLOOR_DIV);
  auto v1 = h(kSym1);
  auto x = h(x_shape);
  return ((x + padding - (kernel - v1) * dilation - v1) / stride) + v1;
}

class OPS_API Conv2D : public Conv {
 public:
  using Conv::Conv;
  ~Conv2D() override = default;
  MS_DECLARE_PARENT(Conv2D, Conv)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr GenOutput(const SymbolPtr &n, const SymbolPtr &h, const SymbolPtr &w) const {
    auto out_channel = input(kIndex1);
    auto format = input_as<StrSymbol>(kIndex7)->value();
    return format == "NCHW" ? ListSymbol::Make({n, out_channel, h, w}) : ListSymbol::Make({n, h, w, out_channel});
  }
};

SymbolPtr Conv2D::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  int64_t pad_mode = ProcessPadMode(input(kIndex3));
  auto format = input_as<StrSymbol>(kIndex7)->value();
  if (!x->HasData()) {
    return GenOutput(GenVInt(), GenVInt(), GenVInt());
  }
  size_t h_axis = kIndex2;
  size_t w_axis = kIndex3;
  if (format == "NHWC") {
    h_axis = kIndex1;
    w_axis = kIndex2;
  }
  auto out_n = x->item(kIndex0);
  SymbolPtr out_h;
  SymbolPtr out_w;
  if (pad_mode == PadMode::VALID) {
    auto kernel = ProcessAttr("Conv2D kernel_size", input(kIndex2), kIndex0, kNum2);
    auto stride = ProcessAttr("Conv2D stride", input(kIndex5), kIndex2, kNum2);
    auto dilation = ProcessAttr("Conv2D dilation", input(kIndex6), kIndex2, kNum2);
    out_h = CalcForPadValid(x->item(h_axis), kernel->item(kIndex0), stride->item(kIndex0), dilation->item(kIndex0));
    out_w = CalcForPadValid(x->item(w_axis), kernel->item(kIndex1), stride->item(kIndex1), dilation->item(kIndex1));
  } else if (pad_mode == PadMode::SAME) {
    auto stride = ProcessAttr("Conv2D stride", input(kIndex5), kIndex2, kNum2);
    out_h = CalcForPadSame(x->item(h_axis), stride->item(kIndex0));
    out_w = CalcForPadSame(x->item(w_axis), stride->item(kIndex1));
  } else if (pad_mode == PadMode::PAD) {
    auto kernel = ProcessAttr("Conv2D kernel_size", input(kIndex2), kIndex0, kNum2);
    auto padding = ProcessAttr("Conv2D pad", input(kIndex4), kIndex0, kNum4);
    auto stride = ProcessAttr("Conv2D stride", input(kIndex5), kIndex2, kNum2);
    auto dilation = ProcessAttr("Conv2D dilation", input(kIndex6), kIndex2, kNum2);
    auto padding_h = Emit(std::make_shared<ScalarAdd>(padding->item(kIndex0), padding->item(kIndex1)));
    auto padding_w = Emit(std::make_shared<ScalarAdd>(padding->item(kIndex2), padding->item(kIndex3)));
    out_h =
      CalcForPadding(x->item(h_axis), kernel->item(kIndex0), padding_h, stride->item(kIndex0), dilation->item(kIndex0));
    out_w =
      CalcForPadding(x->item(w_axis), kernel->item(kIndex1), padding_w, stride->item(kIndex1), dilation->item(kIndex1));
  } else {
    MS_EXCEPTION(NotSupportError) << "The pad_mode " << pad_mode << " is not supported.";
  }
  DoNotEvalOnRun();
  return GenOutput(out_n, out_h, out_w);
}

class OPS_API Conv3D : public Conv {
 public:
  using Conv::Conv;
  ~Conv3D() override = default;
  MS_DECLARE_PARENT(Conv3D, Conv)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr GenOutput(const SymbolPtr &n, const SymbolPtrList &dhw) {
    return GenList({n, input(kIndex1), dhw[kIndex0], dhw[kIndex1], dhw[kIndex2]});
  }
};

SymbolPtr Conv3D::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  int64_t pad_mode = ProcessPadMode(input(kIndex3));
  auto format = input_as<StrSymbol>(kIndex7)->value();
  if (format != "NCDHW") {
    MS_EXCEPTION(NotSupportError) << "Conv3D only support NCDHW format, but got " << format;
  }
  if (!x->HasData()) {
    return GenList({GenVInt(), input(kIndex1), GenVInt(), GenVInt(), GenVInt()});
  }
  SymbolPtrList out_s(kNum3);  // D,H,W
  if (pad_mode == PadMode::VALID) {
    auto kernel = ProcessAttr("Conv3D kernel_size", input(kIndex2), kIndex0, kNum3);
    auto stride = ProcessAttr("Conv3D strides", input(kIndex5), kIndex2, kNum3);
    auto dilation = ProcessAttr("Conv3D dilations", input(kIndex6), kIndex2, kNum3);
    for (size_t i = 0; i < kNum3; i++) {
      out_s[i] = CalcForPadValid(x->item(i + kIndex2), kernel->item(i), stride->item(i), dilation->item(i));
    }
  } else if (pad_mode == PadMode::SAME) {
    auto stride = ProcessAttr("Conv3D strides", input(kIndex5), kIndex2, kNum3);
    for (size_t i = 0; i < kNum3; i++) {
      out_s[i] = CalcForPadSame(x->item(i + kIndex2), stride->item(i));
    }
  } else if (pad_mode == PadMode::PAD) {
    auto kernel = ProcessAttr("Conv3D kernel_size", input(kIndex2), kIndex0, kNum3);
    auto padding = ProcessAttr("Conv3D pad", input(kIndex4), kIndex0, kNum6);
    auto stride = ProcessAttr("Conv3D strides", input(kIndex5), kIndex2, kNum3);
    auto dilation = ProcessAttr("Conv3D dilations", input(kIndex6), kIndex2, kNum3);
    for (size_t i = 0; i < kNum3; i++) {
      auto padding_i = Emit(std::make_shared<ScalarAdd>(padding->item(i * kNum2), padding->item(i * kNum2 + 1)));
      out_s[i] = CalcForPadding(x->item(i + kIndex2), kernel->item(i), padding_i, stride->item(i), dilation->item(i));
    }
  } else {
    MS_EXCEPTION(NotSupportError) << "The pad_mode " << pad_mode << " is not supported.";
  }
  DoNotEvalOnRun();
  return GenList({x->item(kIndex0), input(kIndex1), out_s[kIndex0], out_s[kIndex1], out_s[kIndex2]});
}

REG_SYMBOL_OP_BUILDER("Conv2D")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto out_channel = b->GetInputOrAttr(kIndex3, "out_channel");
    MS_EXCEPTION_IF_NULL(out_channel);
    auto kernel_size = b->GetInputOrAttr(kIndex4, "kernel_size");
    MS_EXCEPTION_IF_NULL(kernel_size);
    auto pad_mode = b->GetInputOrAttr(kIndex6, "pad_mode");
    MS_EXCEPTION_IF_NULL(pad_mode);
    auto padding = b->GetInputOrAttr(kIndex7, "pad");
    MS_EXCEPTION_IF_NULL(padding);
    auto stride = b->GetInputOrAttr(kIndex8, "stride");
    MS_EXCEPTION_IF_NULL(stride);
    auto dilation = b->GetInputOrAttr(kIndex9, "dilation");
    MS_EXCEPTION_IF_NULL(dilation);
    auto format = b->GetInputOrAttr(kIndex11, "format");
    MS_EXCEPTION_IF_NULL(format);
    return b->Emit(std::make_shared<Conv2D>(x, out_channel, kernel_size, pad_mode, padding, stride, dilation, format));
  });

REG_SYMBOL_OP_BUILDER("Conv3D")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto out_channel = b->GetAttr("out_channel");
    MS_EXCEPTION_IF_NULL(out_channel);
    auto kernel_size = b->GetAttr("kernel_size");
    MS_EXCEPTION_IF_NULL(kernel_size);
    auto pad_mode = b->GetAttr("pad_mode");
    MS_EXCEPTION_IF_NULL(pad_mode);
    auto padding = b->GetAttr("pad");
    MS_EXCEPTION_IF_NULL(padding);
    auto stride = b->GetAttr("strides");
    MS_EXCEPTION_IF_NULL(stride);
    auto dilation = b->GetAttr("dilations");
    MS_EXCEPTION_IF_NULL(dilation);
    auto format = b->GetAttr("format");
    MS_EXCEPTION_IF_NULL(format);
    return b->Emit(std::make_shared<Conv3D>(x, out_channel, kernel_size, pad_mode, padding, stride, dilation, format));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
