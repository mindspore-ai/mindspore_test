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
#include "include/backend/debug/execute_order_tracker/execute_order_tracker.h"

#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <iomanip>
#include "include/common/debug/common.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "ir/primitive.h"

namespace mindspore {

namespace {
using ShapeAndType = std::tuple<std::string, std::string, std::string, std::string, size_t, size_t>;

template <typename T>
void WriteCsvFile(const std::string &real_path, const std::vector<T> &data_list,
                  const std::vector<std::pair<std::string, std::function<std::string(const T &)>>> &csv_columns) {
  ChangeFileMode(real_path, S_IRWXU);
  MS_LOG(INFO) << "Dump execute order file path: " << real_path;
  // Clear file contents
  std::ofstream file(real_path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "Failed to open dump execute order file: " << real_path;
  }

  // Writing CSV file header
  for (size_t i = 0; i < csv_columns.size(); ++i) {
    file << csv_columns[i].first;
    if (i < csv_columns.size() - 1) {
      file << ',';
    }
  }
  file << '\n';

  // Writing Data
  for (const auto &data : data_list) {
    for (size_t i = 0; i < csv_columns.size(); ++i) {
      file << std::quoted(csv_columns[i].second(data));
      if (i < csv_columns.size() - 1) {
        file << ',';
      }
    }
    file << '\n';
  }
  file.close();

  // Change the file permissions to read-only
  ChangeFileMode(real_path, S_IRUSR);
}
}  // namespace

bool EnableExecuteOrderDump() {
  // Static variables cache calculation results
  static const bool should_dump = []() {
    const bool is_dry_run = !common::GetEnv(kSimulationLevel).empty();
    return is_dry_run;
  }();
  return should_dump;
}

std::string GetExecuteOrderFilePath(const std::string &file_name) {
  std::string save_path = GetSaveGraphsPathName(file_name);
  auto real_path_opt = Common::CreatePrefixPath(save_path);
  if (!real_path_opt.has_value()) {
    MS_LOG(EXCEPTION) << "Get dump execute order real path failed. PATH=" << save_path;
  }
  return real_path_opt.value();
}

ExecuteOrderTracker &ExecuteOrderTracker::GetInstance() {
  static ExecuteOrderTracker instance;
  return instance;
}

void ExecuteOrderTracker::AddOrderInfo(const OrderInfoPtr &order_info) {
  if (!order_info) {
    MS_LOG(ERROR) << "Null OrderInfoPtr passed to AddOrderInfo.";
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  order_info->index = std::to_string(index_counter_++);
  order_info_list_.emplace_back(order_info);
}

void ExecuteOrderTracker::AddCommOrderInfo(const CommOrderInfoPtr &comm_info) {
  if (!comm_info) {
    MS_LOG(ERROR) << "Null CommOrderInfoPtr passed to AddCommOrderInfo.";
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  comm_order_info_list_.emplace_back(comm_info);
}

void ExecuteOrderTracker::ProcessNode(const CNodePtr &cnode) {
  MS_LOG(INFO) << "The execution order stored program starts running.";
  MS_EXCEPTION_IF_NULL(cnode);
  auto order_info = std::make_shared<OrderInfo>();
  order_info->node_name = cnode->fullname_with_scope();
  order_info->logic_id = std::to_string(AnfAlgo::GetStreamDistinctionLabel(cnode.get()));
  order_info->stream_id = std::to_string(AnfAlgo::GetStreamId(cnode));
  order_info->node_info = cnode->DebugString();
  std::string event_id;
  if (common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode)) {
    event_id = std::to_string(common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrEventId));
  }
  order_info->event_id = event_id;
  std::string group;
  if (common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto group_value = prim->GetAttr(kAttrGroup);
    if (group_value == nullptr) {
      MS_LOG(EXCEPTION) << "Group value is nullptr, node: " << cnode->fullname_with_scope();
    }

    if (group_value->isa<StringImm>()) {
      group = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
    }
  }
  order_info->group = group;

  AddOrderInfo(order_info);
  if (IsCommunicationOp(cnode)) {
    auto comm_info = std::make_shared<CommOrderInfo>();
    comm_info->index = order_info->index;
    comm_info->group = order_info->group;
    // Get comm_ranks and comm_rank strings
    auto comm_ranks = GetCommRanks(order_info->group);
    std::string comm_ranks_str = std::accumulate(
      comm_ranks.begin(), comm_ranks.end(), std::string(),
      [](const std::string &a, uint32_t b) { return a.empty() ? std::to_string(b) : a + " " + std::to_string(b); });
    comm_info->comm_rank = comm_ranks_str;

    auto prim = GetValueNode<PrimitivePtr>(cnode->input(kIndex0));
    MS_EXCEPTION_IF_NULL(prim);
    comm_info->primitive = prim->ToString();

    // Get src_rank, dest_rank, root_rank
    auto [src_rank_str, dest_rank_str, root_rank_str] = GetCommunicationRanks(cnode, comm_ranks);
    comm_info->src_rank = src_rank_str;
    comm_info->dest_rank = dest_rank_str;
    comm_info->root_rank = root_rank_str;

    // Get the shape and type of input and output
    auto [input_shape_str, input_type_str, output_shape_str, output_type_str, input_size, output_size] =
      GetInputOutputShapeAndType(cnode);

    comm_info->input_shape = input_shape_str;
    comm_info->input_type = input_type_str;
    comm_info->output_shape = output_shape_str;
    comm_info->output_type = output_type_str;
    comm_info->input_size = std::to_string(input_size);
    comm_info->output_size = std::to_string(output_size);

    AddCommOrderInfo(comm_info);

    // Cache the path when launching for the first time to avoid losing rank information when destroying
    if (order_path_.empty()) {
      order_path_ = GetExecuteOrderFilePath(kExecuteOrderFileName);
    }

    if (comm_order_path_.empty()) {
      comm_order_path_ = GetExecuteOrderFilePath(kCommExecuteOrderFileName);
    }
  }
  MS_LOG(INFO) << "Execution order storage program runs to the end.";
}

bool ExecuteOrderTracker::IsCommunicationOp(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  return (AnfAlgo::GetKernelType(cnode) == HCCL_KERNEL && common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode));
}

std::vector<uint32_t> ExecuteOrderTracker::GetCommRanks(const std::string &group_name) {
  auto it = comm_ranks_cache_.find(group_name);
  if (it != comm_ranks_cache_.end()) {
    return it->second;
  }

  // If the cache misses, calculate comm_ranks
  std::vector<uint32_t> comm_ranks;
  if (group_name == "hccl_world_group") {
    uint32_t rank_size = 1;

#if !defined(BUILD_LITE)
    rank_size = distributed::collective::CollectiveManager::instance()->global_rank_size();
#endif

    comm_ranks.resize(rank_size);
    std::iota(comm_ranks.begin(), comm_ranks.end(), 0);
  } else {
#if !defined(BUILD_LITE)
    comm_ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(group_name);
#else
    comm_ranks = {0};
#endif
  }

  comm_ranks_cache_[group_name] = comm_ranks;

  return comm_ranks;
}

std::tuple<std::string, std::string, std::string> ExecuteOrderTracker::GetCommunicationRanks(
  const CNodePtr &cnode, const std::vector<uint32_t> &comm_ranks) const {
  MS_EXCEPTION_IF_NULL(cnode);

  auto get_rank_str = [&](const std::string &attr_name) -> std::string {
    uint32_t rank_value = std::numeric_limits<uint32_t>::max();
    if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
      int64_t rank_attr = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr_name);
      if (rank_attr >= 0 && static_cast<size_t>(rank_attr) < comm_ranks.size()) {
        rank_value = comm_ranks[static_cast<size_t>(rank_attr)];
      } else {
        MS_LOG(EXCEPTION) << "Invalid rank_attr value: " << rank_attr << ", or out of range for comm_ranks with size "
                          << comm_ranks.size();
      }
    }
    return rank_value == std::numeric_limits<uint32_t>::max() ? "" : std::to_string(rank_value);
  };
  std::string src_rank = get_rank_str(kAttrSrcRank);
  std::string dest_rank = get_rank_str(kAttrDestRank);
  std::string root_rank = get_rank_str(kAttrRootRank);

  return {src_rank, dest_rank, root_rank};
}

ShapeAndType ExecuteOrderTracker::GetInputOutputShapeAndType(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);

  auto GetShapeAndType = [](AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->GetShapeTrack();
    std::string shape_str = shape ? shape->ToString() : "UnknownShape";

    std::string type_str = "UnknownType";
    auto abs_tensor = abs->cast<abstract::AbstractTensorPtr>();
    const auto &element_abs = abs_tensor ? abs_tensor->element() : nullptr;
    if (element_abs) {
      auto dtype = element_abs->BuildType();
      MS_EXCEPTION_IF_NULL(dtype);
      type_str = dtype->ToString();
    } else {
      auto type = abs->BuildType();
      MS_EXCEPTION_IF_NULL(type);
      type_str = type->ToString();
    }
    return std::make_tuple(shape_str, type_str, abs_tensor);
  };

  auto CalculateSize = [](const abstract::AbstractTensorPtr &abs_tensor) -> size_t {
    if (!abs_tensor) {
      return 0;
    }

    auto base_shape = abs_tensor->GetShape();
    MS_EXCEPTION_IF_NULL(base_shape);

    auto shape = base_shape->cast<abstract::ShapePtr>();
    if (!shape) {
      return 0;
    }

    auto data_size = SizeOf(shape->shape());
    if (data_size == 0) {
      return 0;
    }

    const auto &element_abs = abs_tensor->element();
    MS_EXCEPTION_IF_NULL(element_abs);
    auto dtype = element_abs->BuildType();
    MS_EXCEPTION_IF_NULL(dtype);
    auto type_id = dtype->type_id();
    auto type_size = abstract::TypeIdSize(type_id);
    if (type_size == 0) {
      return 0;
    }
    return data_size * type_size;
  };

  auto input_abs = cnode->input(1)->abstract();
  auto output_abs = cnode->abstract();
  if (!input_abs || !output_abs) {
    return std::make_tuple("UnknownShape", "UnknownType", "UnknownShape", "UnknownType", 0, 0);
  }

  auto [input_shape_str, input_type_str, input_abs_tensor] = GetShapeAndType(input_abs);
  auto [output_shape_str, output_type_str, output_abs_tensor] = GetShapeAndType(output_abs);

  size_t input_size = CalculateSize(input_abs_tensor);
  size_t output_size = CalculateSize(output_abs_tensor);

  return {input_shape_str, input_type_str, output_shape_str, output_type_str, input_size, output_size};
}

void ExecuteOrderTracker::Clear() {
  if (!EnableExecuteOrderDump()) {
    return;
  }

  // If there is no data, return directly
  if (order_info_list_.empty() && comm_order_info_list_.empty()) {
    return;
  }

  // Define CSV output columns and corresponding data acquisition functions
  static const std::vector<std::pair<std::string, std::function<std::string(const OrderInfoPtr &)>>> order_info_csv = {
    {"index", [](const OrderInfoPtr &info) { return info->index; }},
    {"node_name", [](const OrderInfoPtr &info) { return info->node_name; }},
    {"logic_id", [](const OrderInfoPtr &info) { return info->logic_id; }},
    {"stream_id", [](const OrderInfoPtr &info) { return info->stream_id; }},
    {"node_info", [](const OrderInfoPtr &info) { return info->node_info; }},
    {"event_id", [](const OrderInfoPtr &info) { return info->event_id; }},
    {"group", [](const OrderInfoPtr &info) { return info->group; }},
  };

  static const std::vector<std::pair<std::string, std::function<std::string(const CommOrderInfoPtr &)>>>
    comm_order_csv = {
      {"index", [](const CommOrderInfoPtr &info) { return info->index; }},
      {"group", [](const CommOrderInfoPtr &info) { return info->group; }},
      {"comm_rank", [](const CommOrderInfoPtr &info) { return info->comm_rank; }},
      {"primitive", [](const CommOrderInfoPtr &info) { return info->primitive; }},
      {"src_rank", [](const CommOrderInfoPtr &info) { return info->src_rank; }},
      {"dest_rank", [](const CommOrderInfoPtr &info) { return info->dest_rank; }},
      {"root_rank", [](const CommOrderInfoPtr &info) { return info->root_rank; }},
      {"input_shape", [](const CommOrderInfoPtr &info) { return info->input_shape; }},
      {"input_type", [](const CommOrderInfoPtr &info) { return info->input_type; }},
      {"output_shape", [](const CommOrderInfoPtr &info) { return info->output_shape; }},
      {"output_type", [](const CommOrderInfoPtr &info) { return info->output_type; }},
      {"input_size", [](const CommOrderInfoPtr &info) { return info->input_size; }},
      {"output_size", [](const CommOrderInfoPtr &info) { return info->output_size; }},
    };

  // Writing to a CSV file
  if (!order_info_list_.empty()) {
    if (order_path_.empty()) {
      MS_LOG(WARNING) << "Execute order dump path is empty, can not dump execute order.";
      return;
    }
    WriteCsvFile(order_path_, order_info_list_, order_info_csv);
  }

  if (!comm_order_info_list_.empty()) {
    if (comm_order_path_.empty()) {
      MS_LOG(WARNING) << "Comm execute order dump path is empty, can not dump comm execute order.";
      return;
    }
    WriteCsvFile(comm_order_path_, comm_order_info_list_, comm_order_csv);
  }

  MS_LOG(INFO) << "ExecuteOrderTracker data dumped successfully.";

  // Cleaning the data
  order_info_list_.clear();
  comm_order_info_list_.clear();
  comm_ranks_cache_.clear();
  index_counter_ = 1;
}

}  // namespace mindspore
