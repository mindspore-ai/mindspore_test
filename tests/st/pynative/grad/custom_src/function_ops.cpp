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

#include <string>
#include "ms_extension.h"

namespace mindspore::pynative {
namespace autograd {
ShapeVector BroadcastInferShape(const BaseTensorPtr &t1, const BaseTensorPtr &t2) {
  ShapeVector s1 = t1->shape();
  ShapeVector s2 = t2->shape();
  ShapeVector out_shape(std::max(s1.size(), s2.size()), 1LL);
  if (out_shape.empty()) {
    return out_shape;
  }
  for (size_t i = out_shape.size(); i > 0; i--) {
    if (i <= s1.size() && s1[s1.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s1[s1.size() - i];
    } else if (i <= s2.size() && s2[s2.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s2[s2.size() - i];
    }
  }
  return out_shape;
}

class CustomAdd : public Function<CustomAdd> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnAdd", {x, y, MakeValue<int64_t>(1)}, {output});
    return output;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, dout};
  }
};

class CustomIndex : public Function<CustomIndex> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr x, const BaseTensorPtrList &index) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), ShapeVector({1}));
    std::vector<ValuePtr> values;
    for (const auto &item : index) {
      (void)values.emplace_back(item->cast<ValuePtr>());
    }
    auto tuple_index = std::make_shared<ValueTuple>(values);
    custom::CustomLaunchAclnn("aclnnIndex", {x, tuple_index}, {output});
    return output;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, nullptr};
  }
};

class CustomBroadcastTo : public Function<CustomBroadcastTo> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr x, const std::vector<int64_t> &shape) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), shape);
    std::vector<ValuePtr> values;
    for (const auto &item : shape) {
      (void)values.emplace_back(MakeValue<int64_t>(item));
    }
    auto tuple_index = std::make_shared<ValueTuple>(values);
    custom::CustomLaunchAclnn("aclnnExpand", {x, tuple_index}, {output});
    return output;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, nullptr};
  }
};

class CustomMul : public Function<CustomMul> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});
    bool x_require_grad = ctx->NeedGrad(x);
    bool y_require_grad = ctx->NeedGrad(y);
    if (x_require_grad || y_require_grad) {
      ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
    }
    return output;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];

    BaseTensorPtr grad_x = nullptr;
    BaseTensorPtr grad_y = nullptr;

    if (ctx->NeedsInputGrad(0)) {
      grad_x = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[0]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[0]}, {grad_x});
    }
    if (ctx->NeedsInputGrad(1)) {
      grad_y = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[1]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[1]}, {grad_y});
    }

    return {grad_x, grad_y};
  }
};

class CustomMulNoGradOutput : public Function<CustomMulNoGradOutput> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    auto out = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {out});
    ctx->MarkNonDifferentiable({out});
    return out;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_output) {
    return {grad_output[0], grad_output[0]};
  }
};

class CustomMulNoGradInput : public Function<CustomMulNoGradInput> {
 public:
  static BaseTensorPtrList Forward(AutogradContext *ctx, const BaseTensorPtr &input) {
    auto out = std::make_shared<BaseTensor>(input->data_type(), input->shape());
    custom::CustomLaunchAclnn("aclnnMul", {input, input}, {out});
    ctx->MarkNonDifferentiable({input});
    return {input, out};
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_output) {
    return {grad_output[0], ctx->NeedsInputGrad(0) ? grad_output[0] : nullptr};
  }
};

class CustomInplaceMulOp : public Function<CustomInplaceMulOp> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {x});
    ctx->MarkDirty({x});
    if (ctx->NeedGrad(x)) {
      ctx->SaveForBackward({y});
    }
    return x;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_output) {
    BaseTensorPtr grad_input = nullptr;
    if (ctx->NeedsInputGrad(0)) {
      auto saved_tensor = ctx->GetSavedTensors();
      grad_input =
        std::make_shared<BaseTensor>(grad_output[0]->data_type(), BroadcastInferShape(grad_output[0], saved_tensor[0]));
      custom::CustomLaunchAclnn("aclnnMul", {grad_output[0], saved_tensor[0]}, {grad_input});
    }
    return {grad_input};
  }
};

BaseTensorPtr run_custom_add(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomAdd::Apply(x, y);
}

BaseTensorPtr run_custom_index(const BaseTensorPtr x, const BaseTensorPtrList &index) {
  return CustomIndex::Apply(x, index);
}

BaseTensorPtr run_custom_broadcast_to(const BaseTensorPtr x, const std::vector<int64_t> &index) {
  return CustomBroadcastTo::Apply(x, index);
}

BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}

BaseTensorPtr run_inplace_mul_op(const tensor::BaseTensorPtr &input, const tensor::BaseTensorPtr &other) {
  return CustomInplaceMulOp::Apply(input, other);
}

BaseTensorPtr run_mul_mark_no_diff_output(const tensor::BaseTensorPtr &input, const tensor::BaseTensorPtr &other) {
  return CustomMulNoGradOutput::Apply(input, other);
}

BaseTensorPtrList run_mul_mark_no_diff_input(const tensor::BaseTensorPtr &input) {
  return CustomMulNoGradInput::Apply(input);
}
}  // namespace autograd
}  // namespace mindspore::pynative

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("add", &mindspore::pynative::autograd::run_custom_add, "out = x + y");
  m.def("index", &mindspore::pynative::autograd::run_custom_index, "tensor[index]");
  m.def("broadcast_to", &mindspore::pynative::autograd::run_custom_broadcast_to, "shape1->shape2");
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "out = x * y");
  m.def("inplace_mul", &mindspore::pynative::autograd::run_inplace_mul_op, "x = x * y");
  m.def("mul_no_diff_out", &mindspore::pynative::autograd::run_mul_mark_no_diff_output, "out = x * y, no diff output");
  m.def("mul_no_diff_in", &mindspore::pynative::autograd::run_mul_mark_no_diff_input, "out = x * y, no diff input");
}
