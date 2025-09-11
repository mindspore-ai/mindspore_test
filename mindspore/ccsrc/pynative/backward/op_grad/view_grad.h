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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_

#include <vector>
#include <string>
#include <utility>

#include "tools/profiler/profiler.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "pyboost/functions/auto_grad_guard.h"
#include "pynative/backward/grad_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore::pynative::autograd {
template <typename Func>
void DoViewGrad(const TensorPtr &input_tensor, const TensorPtr &output_tensor, const Func &make_func,
                bool is_safe = true) {
  static const std::string kDoGradName = "DoViewGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);
  const bool requires_grad = kernel::pyboost::OpRunStatus::Get().RequireGrad();
  is_safe ? AutoGradUtil::MakeOutput(requires_grad, output_tensor, input_tensor)
          : AutoGradUtil::MakeOutput(requires_grad, output_tensor);

  if (requires_grad) {
    runtime::Pipeline::Get().WaitBpropStage();
    if (AutoGradUtil::NeedGrad(input_tensor)) {
      auto view_grad_node = make_func();
      UpdateNextEdges(view_grad_node, {input_tensor});
      auto output_meta_data = output_tensor->auto_grad_meta_data();
      output_meta_data->set_grad_node(view_grad_node);
    }
    UpdateVersion(output_tensor);
  }
}

class ViewBackwardNode : public BackwardNode {
 public:
  ViewBackwardNode(std::string name, std::vector<int64_t> self_shape)
      : BackwardNode(std::move(name)), self_shape_(std::move(self_shape)) {}
  ~ViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
};

class TransposeBackwardNode : public BackwardNode {
 public:
  TransposeBackwardNode(std::string name, std::vector<int64_t> perm)
      : BackwardNode(std::move(name)), perm_(std::move(perm)) {}
  ~TransposeBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> perm_;
};

class TransposeExtViewBackwardNode : public BackwardNode {
 public:
  TransposeExtViewBackwardNode(std::string name, int64_t dim0, int64_t dim1)
      : BackwardNode(std::move(name)), dim0_(dim0), dim1_(dim1) {}
  ~TransposeExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  int64_t dim0_;
  int64_t dim1_;
};

class SelectExtViewBackwardNode : public BackwardNode {
 public:
  SelectExtViewBackwardNode(std::string name, std::vector<int64_t> self_shape, int64_t dim, int64_t index)
      : BackwardNode(std::move(name)),
        self_shape_(std::move(self_shape)),
        dim_(std::move(dim)),
        index_(std::move(index)) {}
  ~SelectExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  int64_t dim_;
  int64_t index_;
};

class SliceExtViewBackwardNode : public BackwardNode {
 public:
  SliceExtViewBackwardNode(std::string name, std::vector<int64_t> self_shape, int64_t dim, int64_t start, int64_t end,
                           int64_t step)
      : BackwardNode(std::move(name)),
        self_shape_(std::move(self_shape)),
        dim_(std::move(dim)),
        start_(std::move(start)),
        end_{std::move(end)},
        step_{std::move(step)} {}
  ~SliceExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t step_;
};
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_
