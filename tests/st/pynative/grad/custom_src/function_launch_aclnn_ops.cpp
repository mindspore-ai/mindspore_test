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

#include <string>
#include "ms_extension.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/ascend/opapi/aclnn/custom_aclnn_utils.h"
#include "runtime/pynative/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"

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

class CustomMul : public Function<CustomMul> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));

    auto p = std::make_shared<Primitive>("CustomLaunchAclnn");
    auto op = std::make_shared<kernel::pyboost::OpRunner>(p, runtime::OpRunner::GetDeviceContext("Ascend"));
    op->set_stream_id(kernel::pyboost::PyBoostUtils::cur_stream_id());
    op->set_outputs({output});
    // No need to convert input
    kernel::pyboost::PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, y);
    kernel::pyboost::PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

    // Async
    kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x, y]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      kernel::pyboost::PyBoostUtils::MallocOpInputs(device_context, x, y);
      // Malloc for output tensors
      kernel::pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnMul, device_context, op->stream_id(), x, y, outputs[0]);
      MS_LOG(DEBUG) << "Run device task Add end";
    }));

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

BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}

}  // namespace autograd
}  // namespace mindspore::pynative

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "out = x * y"); }
