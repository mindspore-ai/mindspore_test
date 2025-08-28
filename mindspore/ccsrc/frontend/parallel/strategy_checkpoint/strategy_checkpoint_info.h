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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "nlohmann/json.hpp"
#include "utils/hash_map.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "proto/node_strategy.pb.h"

namespace mindspore {
namespace parallel {
using StrategyMap = mindspore::HashMap<std::string, StrategyPtr>;
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
using TensorInfoMap = mindspore::HashMap<std::string, TensorLayoutPtr>;
using TensorLayoutMap = mindspore::HashMap<std::string, std::vector<TensorLayoutPtr>>;
using ParameterMap = std::vector<std::pair<std::string, ParameterPtr>>;
using ManualShapeMap = mindspore::HashMap<std::string, std::vector<std::pair<int64_t, int64_t>>>;
using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
using TensorLayoutValueMap = mindspore::HashMap<std::string, ValueTuplePtr>;

class StrategyCheckpointInfo {
 public:
  StrategyCheckpointInfo() : current_stage_(0) {}
  virtual ~StrategyCheckpointInfo() = default;
  void Init(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
            const ManualShapeMap &manual_shape_map, int64_t current_stage) {
    strategy_map_ = strategy_map;
    out_strategy_map_ = StrategyMap();
    tensor_info_map_ = tensor_info_map;
    manual_shape_map_ = manual_shape_map;
    current_stage_ = current_stage;
  }
  StrategyMap strategy_map() const { return strategy_map_; }
  void set_strategy_map(const StrategyMap &strategy_map);
  StrategyMap out_strategy_map() const { return out_strategy_map_; }
  void set_out_strategy_map(const StrategyMap &out_strategy_map);
  TensorInfoMap tensor_info_map() const { return tensor_info_map_; }
  void set_tensor_info_map(const TensorInfoMap &tensor_info_map);
  ManualShapeMap manual_shape_map() const { return manual_shape_map_; }
  void set_manual_shape_map(const ManualShapeMap &manual_shape_map);
  int64_t current_stage() const { return current_stage_; }
  TensorLayoutValueMap tensor_layout_map() const { return tensor_layout_map_; }
  void set_tensor_layout_map(const TensorLayoutValueMap &tensor_layout_map);
  TensorLayoutValueMap out_tensor_layout_map() const { return out_tensor_layout_map_; }
  void set_out_tensor_layout_map(const TensorLayoutValueMap &out_tensor_layout_map);
  TensorLayoutValueMap tensor_layout_newshape_map() const { return tensor_layout_newshape_map_; }
  void set_tensor_layout_newshape_map(const TensorLayoutValueMap &tensor_layout_newshape_map);
  TensorLayoutValueMap out_tensor_layout_newshape_map() const { return out_tensor_layout_newshape_map_; }
  void set_out_tensor_layout_newshape_map(const TensorLayoutValueMap &out_tensor_layout_newshape_map_);

  virtual void FromJson(const nlohmann::json &stra_ckpt_info_j);
  nlohmann::json to_json() const;
  nlohmann::json to_json_strategy_item(const StrategyPtr &stra_pair) const;
  nlohmann::json to_json_tensorinfo_item(const std::string &parameter_name, const TensorLayoutPtr &layout) const;
  nlohmann::json to_json_layout_value_tuple_item(const std::string &node_name,
                                                 const ValueTuplePtr &layout_value_tuple) const;
  void from_protobuf(const straspb::ParallelStrategyMap &parallel_strategy_map);
  straspb::ParallelStrategyMap to_protobuf() const;

 protected:
  StrategyMap strategy_map_;
  StrategyMap out_strategy_map_;
  int64_t current_stage_;
  TensorInfoMap tensor_info_map_;
  TensorInfoMap out_tensor_info_map_;
  ManualShapeMap manual_shape_map_;
  TensorLayoutValueMap tensor_layout_map_;
  TensorLayoutValueMap out_tensor_layout_map_;
  TensorLayoutValueMap tensor_layout_newshape_map_;
  TensorLayoutValueMap out_tensor_layout_newshape_map_;
};

class StrategyJsonInfo : public StrategyCheckpointInfo {
 public:
  StrategyJsonInfo() : StrategyCheckpointInfo() {}
  ~StrategyJsonInfo() override = default;
  void Init(const StrategyMap &strategy_map, const StrategyMap &out_strategy_map,
            const TensorLayoutValueMap &tensor_layout_map, const TensorLayoutValueMap &out_tensor_layout_map,
            const TensorLayoutValueMap &tensor_layout_newshape_map,
            const TensorLayoutValueMap &out_tensor_layout_newshape_map, int64_t current_stage) {
    strategy_map_ = strategy_map;
    out_strategy_map_ = out_strategy_map;
    tensor_layout_map_ = tensor_layout_map;
    out_tensor_layout_map_ = out_tensor_layout_map;
    tensor_layout_newshape_map_ = tensor_layout_newshape_map;
    out_tensor_layout_newshape_map_ = out_tensor_layout_newshape_map;
    tensor_info_map_ = TensorInfoMap();
    manual_shape_map_ = ManualShapeMap();
    current_stage_ = current_stage;
  }
  void FromJson(const nlohmann::json &stra_json_info_j) override;
  void StrategyFromJson(const nlohmann::json &stra_json_info_j);
  void LayoutFromJson(const nlohmann::json &stra_json_info_j);
  void NewShapeLayoutFromJson(const nlohmann::json &stra_json_info_j);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_STRATEGY_CHECKPOINT_INFO_H_
