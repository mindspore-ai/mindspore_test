/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_

#include <vector>
#include <memory>
#include <string>

#include "frontend/parallel/ops_info/reduce_method_info.h"

namespace mindspore {
namespace parallel {
class ReduceBaseMethod : public ReduceMethod {
 public:
  ReduceBaseMethod(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ReduceBaseMethod() override = default;

 protected:
  Status InferMirrorOps() override;
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
};

class ReduceMaxInfo : public ReduceBaseMethod {
 public:
  ReduceMaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMaxCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  ~ReduceMaxInfo() override = default;
};

class ReduceMeanInfo : public ReduceBaseMethod {
 public:
  ReduceMeanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMeanCost>()) {
    reduce_method_ = REDUCE_OP_MEAN;
  }

  ~ReduceMeanInfo() override = default;

 protected:
  Status InferForwardCommunication() override;
};

class MeanExtInfo : public ReduceMeanInfo {
 public:
  MeanExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ReduceMeanInfo(name, inputs_shape, outputs_shape, attrs) {}

  ~MeanExtInfo() override = default;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status CheckInputLayout() override;
  Status InferOutputTensorInfo() override;
  Status CheckOutputLayout() override;
  Status InferForwardCommunicationByLayout() override;

 private:
  bool is_infer_out_layout_ = false;
};

class ReduceSumInfo : public ReduceBaseMethod {
 public:
  ReduceSumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_SUM;
  }

  ~ReduceSumInfo() override = default;
};

class ReduceAnyInfo : public ReduceBaseMethod {
 public:
  ReduceAnyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_ANY;
  }
  ~ReduceAnyInfo() override = default;

 protected:
  Status InferForwardCommunication() override;
  ForwardOp CreateForwardOp(const std::vector<Group> &forward_group) const;
};

class ReduceMinInfo : public ReduceBaseMethod {
 public:
  ReduceMinInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMinCost>()) {
    reduce_method_ = REDUCE_OP_MIN;
  }

  ~ReduceMinInfo() override = default;
};

class ReduceProdInfo : public ReduceBaseMethod {
 public:
  ReduceProdInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceProdCost>()) {
    reduce_method_ = REDUCE_OP_PROD;
  }

  ~ReduceProdInfo() override = default;
};

class ReduceAllInfo : public ReduceAnyInfo {
 public:
  ReduceAllInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceAnyInfo(name, inputs_shape, outputs_shape, attrs) {
    reduce_method_ = REDUCE_OP_ALL;
  }

  ~ReduceAllInfo() override = default;
};

class SumExtInfo : public ReduceBaseMethod {
 public:
  SumExtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_SUM;
  }

  ~SumExtInfo() override = default;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status CheckInputLayout() override;
  Status InferOutputTensorInfo() override;
  Status CheckOutputLayout() override;
  Status InferForwardCommunicationByLayout() override;

 private:
  bool is_infer_out_layout_ = false;
};

class MaxInfo : public ReduceMaxInfo {
 public:
  MaxInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape, const PrimitiveAttrs &attrs)
      : ReduceMaxInfo(name, input_shape, output_shape, attrs) {}
  ~MaxInfo() = default;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  std::vector<int64_t> reduce_dim() override;
  Status CheckInputLayout() override;
  Status InferOutputTensorInfo() override;
  Status CheckOutputLayout() override;
  Status InferForwardCommunicationByLayout() override;

 private:
  // Check if the output layout is derived by the framework based on the input layout
  bool is_infer_out_layout_ = false;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_
