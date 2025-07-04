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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MOE_F_F_N_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MOE_F_F_N_INFO_H_

#include <ir/value.h>
#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class FFNInfo : public OperatorInfo {
 public:
  FFNInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
          const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ActivationInfoCost>()) {}
  ~FFNInfo() override = default;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status GetAttrs() override { return SUCCESS; };
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferDevMatrixShape() override;
  Status SetAntiquantTensorMap(int64_t antiquant_index, const Shape &antiquant_tensor_map,
                               const Shape &antiquant_group_tensor_map);
  Status InferAntiQuantTensorMap(int64_t expert_pos);
  void InitInputsExist();
  bool IsInputExist(size_t index);
  size_t GetStrategyRealIndex(size_t index);

 private:
  Shape origin_dev_matrix_shape_;
  std::vector<bool> inputs_exist_;
};

class FFNExtInfo : public FFNInfo {
 public:
  FFNExtInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : FFNInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~FFNExtInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MOE_F_F_N_INFO_H_
