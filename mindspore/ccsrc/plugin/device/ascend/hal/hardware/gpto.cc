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
#include "plugin/device/ascend/hal/hardware/gpto.h"

#include <cmath>
#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <stack>
#include <tuple>

#include "op_def/math_ops.h"
#include "op_def/conv_pool_op_name.h"
#include "op_def/ascend_op_name.h"
#include "op_def/other_op_name.h"
#include "utils/anf_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/misc.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_memory_adapter.h"
#include "runtime/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace gpto {
bool Overlap(const Time &start1, const Time &end1, const Time &start2, const Time &end2) {
  return (start1 >= start2 && start1 < end2) ||
         (start2 >= start1 && start2 < end1);  // if equal start and end for two intervals, then no overlap
}

std::pair<bool, std::string> GetDebugConfig() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto enable_save_graphs =
    (context_ptr->CanDump(kIntroductory)) || (common::GetEnv("MS_ENABLE_GPTO_VERIFICATION") == "1");
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  return std::make_pair(enable_save_graphs, save_graphs_path);
}

size_t ScheduleToEvents(const SchedulingOutput &schedule) {
  size_t count = 0;

  std::set<Interval, SortByStart> tasks_start;
  std::set<Interval, SortByEndMin> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_start.insert(task_time);
    tasks_end.insert(task_time);
  }

  for (auto it = tasks_start.begin(); it != tasks_start.end(); ++it) {
    tasks_end.erase(*it);
    // Dismiss overlapping tasks: save min end value of non-overlapping task to the right
    std::unordered_map<GptoTaskPtr, bool> dismissed;
    auto it1 = std::next(it);
    for (; Overlap(it->start, it->end, it1->start, it1->end) && it1 != tasks_start.end(); ++it1) {
      dismissed[it1->task] = true;
    }
    Time min_end_value = 0;
    for (auto it2 = tasks_end.begin(); it2 != tasks_end.end(); ++it2) {
      if (!dismissed[it2->task]) {
        min_end_value = it2->end;
        break;
      }
    }
    // Add events to immediate right neighborhood
    for (; it1->start < min_end_value && it1 != tasks_start.end(); ++it1) {
      if (it->task->gpto_type() != it1->task->gpto_type()) {
        it1->task->recv_events().insert(it->task);
        MS_LOG(DEBUG) << "GPTO scheduling event from " << it->task->id() << " to " << it1->task->id();
        count++;
      }
    }
  }
  MS_LOG(INFO) << "Generated " << count << " events to impose GPTO schedule";
  return count;
}

size_t MemorySafetyEvents(const SchedulingOutput &schedule) {
  size_t count = 0;

  std::set<Interval, SortByEndMax> tasks_end;
  for (const auto &task_time : schedule.task_times) {
    tasks_end.insert(task_time);
  }

  for (auto it = tasks_end.begin(); it != tasks_end.end(); ++it) {
    if (it->task->gpto_type() == 0) {
      continue;
    }
    for (auto it1 = std::next(it); it1 != tasks_end.end(); ++it1) {
      if (Overlap(it->start, it->end, it1->start, it1->end)) {
        continue;
      }
      if (it1->task->gpto_type() == 0) {
        if (it->task->recv_events().find(it1->task) == it->task->recv_events().end()) {
          it->task->recv_events().insert(it1->task);
          count++;
          MS_LOG(DEBUG) << "Memory safety event from " << it1->task->id() << " to " << it->task->id();
        }
        break;
      }
    }
  }
  MS_LOG(INFO) << "Generated " << count << " events to ensure dynamic memory safety";
  return count;
}

// Sorting for tasks
bool SortByCostMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->cost() > task2->cost() || (task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByCostMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->cost() < task2->cost() || (task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortBySuccDiff(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->succ_diff_type() > task2->succ_diff_type() ||
           (task1->succ_diff_type() == task2->succ_diff_type() && task1->cost() > task2->cost()) ||
           (task1->succ_diff_type() == task2->succ_diff_type() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByBottomLevelMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() > task2->bottom_level() ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() > task2->cost()) ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByBottomLevelMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() < task2->bottom_level() ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() > task2->cost()) ||
           (task1->bottom_level() == task2->bottom_level() && task1->cost() == task2->cost() &&
            task1->id() < task2->id())));
}

bool SortByTopLevelMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() > task2->top_level() ||
           (task1->top_level() == task2->top_level() && task1->cost() > task2->cost()) ||
           (task1->top_level() == task2->top_level() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByTopLevelMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() < task2->top_level() ||
           (task1->top_level() == task2->top_level() && task1->cost() > task2->cost()) ||
           (task1->top_level() == task2->top_level() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelMaxSum(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() + task1->bottom_level() > task2->top_level() + task2->bottom_level() ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() > task2->cost()) ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelMinSum(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->top_level() + task1->bottom_level() < task2->top_level() + task2->bottom_level() ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() > task2->cost()) ||
           (task1->top_level() + task1->bottom_level() == task2->top_level() + task2->bottom_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByBottomTopLevelComposite(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->bottom_level() - task1->top_level() > task2->bottom_level() - task2->top_level() ||
           (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
            task1->cost() > task2->cost()) ||
           (task1->bottom_level() - task1->top_level() == task2->bottom_level() - task2->top_level() &&
            task1->cost() == task2->cost() && task1->id() < task2->id())));
}

bool SortByWeightedLength(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->weighted_length() > task2->weighted_length() ||
           (task1->weighted_length() == task2->weighted_length() && task1->id() < task2->id())));
}

bool SortByDepthMax(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->depth() > task2->depth() || (task1->depth() == task2->depth() && task1->cost() > task2->cost()) ||
           (task1->depth() == task2->depth() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

// BFS with costs for tie breaking
bool SortByDepthMin(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->depth() < task2->depth() || (task1->depth() == task2->depth() && task1->cost() > task2->cost()) ||
           (task1->depth() == task2->depth() && task1->cost() == task2->cost() && task1->id() < task2->id())));
}

// Sort by predecessor to comm
bool SortByPredComm(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

// Sort by predecessor to comm + DFS
bool SortByPredCommDepth(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_comm() < task2->pred_comm() ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() > task2->depth()) ||
           (task1->pred_comm() == task2->pred_comm() && task1->depth() == task2->depth() &&
            task1->id() < task2->id())));
}

// Sort by predecessor to cube + bottom level
bool SortByPredCube(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->pred_cube() < task2->pred_cube() ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() > task2->bottom_level()) ||
           (task1->pred_cube() == task2->pred_cube() && task1->bottom_level() == task2->bottom_level() &&
            task1->id() < task2->id())));
}

// Sort by greedy height of memory (maintained dynamically)
bool SortByGreedyHeight(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() && (task1->initial_mem_impact() - task1->minus_mem_impact() <
                                                             task2->initial_mem_impact() - task2->minus_mem_impact() ||
                                                           (task1->initial_mem_impact() - task1->minus_mem_impact() ==
                                                              task2->initial_mem_impact() - task2->minus_mem_impact() &&
                                                            SortByBottomLevelMax(task1, task2))));
}

bool SortByReversePostOrder(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->post_order_time() > task2->post_order_time() ||
           (task1->post_order_time() == task2->post_order_time() && task1->id() < task2->id())));
}

bool SortBySValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->s_value() > task2->s_value() || (task1->s_value() == task2->s_value() && task1->id() < task2->id())));
}

bool SortByAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->a_value() > task2->a_value() || (task1->a_value() == task2->a_value() && task1->id() < task2->id())));
}

bool SortByMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->m_value() > task2->m_value() || (task1->m_value() == task2->m_value() && task1->id() < task2->id())));
}

bool SortByWeightedSValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->sw_value() > task2->sw_value() ||
           (task1->sw_value() == task2->sw_value() && task1->id() < task2->id())));
}

bool SortByWeightedAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->aw_value() > task2->aw_value() ||
           (task1->aw_value() == task2->aw_value() && task1->id() < task2->id())));
}

bool SortByWeightedMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->mw_value() > task2->mw_value() ||
           (task1->mw_value() == task2->mw_value() && task1->id() < task2->id())));
}

bool SortByCostSValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->sc_value() > task2->sc_value() ||
           (task1->sc_value() == task2->sc_value() && task1->id() < task2->id())));
}

bool SortByCostAValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->ac_value() > task2->ac_value() ||
           (task1->ac_value() == task2->ac_value() && task1->id() < task2->id())));
}

bool SortByCostMValue(const GptoTaskPtr &task1, const GptoTaskPtr &task2) {
  return task1->subgraph_id() < task2->subgraph_id() ||
         (task1->subgraph_id() == task2->subgraph_id() &&
          (task1->mc_value() > task2->mc_value() ||
           (task1->mc_value() == task2->mc_value() && task1->id() < task2->id())));
}

// Get PEs description
std::map<GptoTaskType, int32_t> GetPEs([[maybe_unused]] std::map<std::string, size_t> *group_to_id) {
  std::map<GptoTaskType, int32_t> new_pem;
  if (gpto_mode == kComp) {
    new_pem[kVec] = 1;
  } else if (gpto_mode == kCompComm) {
    new_pem[kVec] = 1;
    new_pem[kComm] = 1;
  } else if (gpto_mode == kCompCubeComm) {
    new_pem[kVec] = 1;
    new_pem[kCube] = 1;
    new_pem[kComm] = 1;
  } else if (gpto_mode == kCompCommGroup) {
    new_pem[kVec] = 1;
    for (auto &group : *group_to_id) {
      new_pem[group.second] = 1;
    }
  }
  return new_pem;
}

// Auxiliary subroutines and lower bounds
void ComputeDepthAndTopLevel(const std::vector<GptoTaskPtr> &tasks) {
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::queue<GptoTaskPtr> tasks_to_visit;
  // Initialization loop
  for (size_t j = 0; j < tasks.size(); ++j) {
    const auto &id = tasks[j]->id();
    unprocessed_parents[id] = tasks[j]->parents().size();
    if (unprocessed_parents[id] == 0) {
      tasks[j]->set_top_level(tasks[j]->cost());
      tasks_to_visit.push(tasks[j]);
    }
  }
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      successor->set_depth(std::max(successor->depth(), selected_task->depth() + 1));
      successor->set_top_level(std::max(successor->top_level(), selected_task->top_level() + successor->cost()));
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        tasks_to_visit.push(successor);
      }
    }
    tasks_to_visit.pop();
  }
}

void ComputeBottomLevelAndWeightedLength(const std::vector<GptoTaskPtr> &tasks) {
  std::unordered_map<GptoTaskId, size_t> unprocessed_children;
  std::unordered_map<GptoTaskId, double> children_sum;
  std::unordered_map<GptoTaskId, double> children_max;
  std::queue<GptoTaskPtr> tasks_to_visit;
  // Initialization loop: bottom and weighted_length already initialized to cost when AssignCost() is called
  for (auto &task : tasks) {
    const auto &id = task->id();
    unprocessed_children[id] = task->children().size();
    if (unprocessed_children[id] == 0) {
      tasks_to_visit.push(task);
    }
  }
  while (!tasks_to_visit.empty()) {
    const auto &selected_task = tasks_to_visit.front();
    // Update candidate tasks
    for (auto &predecessor : selected_task->parents()) {
      const auto &pred_id = predecessor.lock()->id();
      predecessor.lock()->set_bottom_level(
        std::max(predecessor.lock()->bottom_level(), selected_task->bottom_level() + predecessor.lock()->cost()));
      children_sum[pred_id] += selected_task->weighted_length();
      children_max[pred_id] = std::max(children_max[pred_id], selected_task->weighted_length());
      unprocessed_children[pred_id] -= 1;
      if (unprocessed_children[pred_id] == 0) {
        if (children_max[pred_id] == 0) {
          MS_LOG(EXCEPTION) << "Divisor children_max[pred_id] cannot be 0!";
        }
        predecessor.lock()->set_weighted_length(predecessor.lock()->cost() + children_max[pred_id] +
                                                children_sum[pred_id] / children_max[pred_id]);
        tasks_to_visit.push(predecessor.lock());
      }
    }
    tasks_to_visit.pop();
  }
}

void ComputePostOrder(const std::vector<GptoTaskPtr> &tasks) {
  std::stack<GptoTaskPtr> tasks_to_visit;
  std::unordered_map<GptoTaskId, bool> visited;

  for (auto &task : tasks) {
    visited[task->id()] = false;
    if (task->parents().size() == 0) {
      tasks_to_visit.push(task);
    }
  }

  size_t step = 0;
  while (!tasks_to_visit.empty()) {
    auto &selected_task = tasks_to_visit.top();
    bool zero_unvisited_children = true;
    for (auto &child : selected_task->children()) {
      if (!visited[child->id()]) {
        tasks_to_visit.push(child);
        zero_unvisited_children = false;
      }
    }
    if (zero_unvisited_children) {
      tasks_to_visit.pop();
      if (!visited[selected_task->id()]) {
        selected_task->set_post_order_time(step++);
        visited[selected_task->id()] = true;
      }
    }
  }
}

void ComputePredComm(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_comm(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() >= kComm) {
        task->set_pred_comm(task->pred_comm() + 1);
      }
    }
  }
}

void ComputePredCube(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    task->set_pred_cube(0);
    for (auto &predecessor : task->parents()) {
      if (predecessor.lock()->gpto_type() == kCube) {
        task->set_pred_cube(task->pred_cube() + 1);
      }
    }
  }
}

void ComputeInitialMemoryImpact(const std::vector<GptoTaskPtr> &tasks) {
  for (auto &task : tasks) {
    Memory out_weight = 0, workspace_weight = 0;
    for (auto &tensor : task->out_tensors()) {
      if (tensor->type() == kWorkspace) {
        workspace_weight += tensor->weight();
      } else {
        out_weight += tensor->weight();
      }
    }
    for (auto &tensor : task->workspace_tensors()) {
      workspace_weight += tensor->weight();
    }
    task->set_workspace_memory(workspace_weight);
    task->set_initial_mem_impact(out_weight + workspace_weight);
    MS_LOG(DEBUG) << "Initial memory impact for task " << task->id() << " is " << task->initial_mem_impact()
                  << ", workspace is " << task->workspace_memory();
  }
}

Time LowerBoundBottomLevel(const std::vector<GptoTaskPtr> &tasks) {
  Time max_bottom_level = 0;
  for (const auto &task : tasks) {
    max_bottom_level = std::max(max_bottom_level, task->bottom_level());
  }
  return max_bottom_level;
}

Time TotalTime(const std::vector<GptoTaskPtr> &tasks) {
  return std::accumulate(tasks.begin(), tasks.end(), static_cast<Time>(0),
                         [](Time acc, const auto &task) { return acc + task->cost(); });
}

Time LowerBoundPEs(const std::vector<GptoTaskPtr> &tasks,
                   const std::map<GptoTaskType, int32_t> &type_to_num_cores_map) {
  double lower_bound = 0;

  std::unordered_map<GptoTaskType, Time> type_task_sum;
  for (const auto &task : tasks) {
    type_task_sum[task->gpto_type()] += task->cost();
  }
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    if (num_cores == 0) {
      MS_LOG(EXCEPTION) << "Divisor num_cores cannot be 0!";
    }
    lower_bound = std::max(lower_bound, type_task_sum[type] / (1.0 * num_cores));
  }
  return std::ceil(lower_bound);
}

constexpr TaskSortFunction TASK_SORT[] = {SortByCostMax,
                                          SortByCostMin,
                                          SortBySuccDiff,
                                          SortByBottomLevelMax,
                                          SortByBottomLevelMin,
                                          SortByTopLevelMax,
                                          SortByTopLevelMin,
                                          SortByBottomTopLevelMaxSum,
                                          SortByBottomTopLevelMinSum,
                                          SortByBottomTopLevelComposite,
                                          SortByWeightedLength,
                                          SortByDepthMax,
                                          SortByDepthMin,
                                          SortByPredComm,
                                          SortByPredCommDepth,
                                          SortByPredCube,
                                          SortByGreedyHeight,
                                          SortBySValue,
                                          SortByAValue,
                                          SortByMValue,
                                          SortByWeightedSValue,
                                          SortByWeightedAValue,
                                          SortByWeightedMValue,
                                          SortByCostSValue,
                                          SortByCostAValue,
                                          SortByCostMValue,
                                          SortByReversePostOrder};

constexpr std::string_view TASK_SORT_NAMES[] = {"SortByCostMax",
                                                "SortByCostMin",
                                                "SortBySuccDiff",
                                                "SortByBottomLevelMax",
                                                "SortByBottomLevelMin",
                                                "SortByTopLevelMax",
                                                "SortByTopLevelMin",
                                                "SortByBottomTopLevelMaxSum",
                                                "SortByBottomTopLevelMinSum",
                                                "SortByBottomTopLevelComposite",
                                                "SortByWeightedLength",
                                                "SortByDepthMax",
                                                "SortByDepthMin",
                                                "SortByPredComm",
                                                "SortByPredCommDepth",
                                                "SortByPredCube",
                                                "SortByGreedyHeight",
                                                "SortBySValue",
                                                "SortByAValue",
                                                "SortByMValue",
                                                "SortByWeightedSValue",
                                                "SortByWeightedAValue",
                                                "SortByWeightedMValue",
                                                "SortByCostSValue",
                                                "SortByCostAValue",
                                                "SortByCostMValue",
                                                "SortByReversePostOrder"};

constexpr std::string_view PE_NAME_SORT[] = {"SortByLoad", "SortByValidStart"};

bool SkipAlgorithm(size_t task_sort, size_t pes_sort) {
  if (common::GetEnv("MS_ENABLE_GPTO_ALGO") != "" &&
      common::GetEnv("MS_ENABLE_GPTO_ALGO") != TASK_SORT_NAMES[task_sort]) {
    return true;
  }

  if (gpto_mode == kComp &&
      (TASK_SORT_NAMES[task_sort] == "SortByPredComm" || TASK_SORT_NAMES[task_sort] == "SortByPredCommDepth")) {
    return true;
  }
  if (gpto_mode != kCompCubeComm && TASK_SORT_NAMES[task_sort] == "SortByPredCube") {
    return true;
  }
  if (pes_sort == static_cast<size_t>(PEsSort::kSortByValidStart)) {  // same solution if 1 PE per type
    return true;
  }
  return false;
}

SchedulingOutput MemAwareScheduler(const SchedulingInput &input,
                                   [[maybe_unused]] std::map<std::string, size_t> *group_to_id) {
  const std::vector<GptoTaskPtr> *tasks = &(input.tasks);
  auto type_to_num_cores_map = GetPEs(group_to_id);
  SchedulingOutput output{{}, SIZE_MAX, MEMORY_LIMIT};
  output.task_times.reserve(input.tasks.size());

  // Optional: verify input task graph is a DAG
  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    if (VerifyDAG(*tasks)) {
      MS_LOG(INFO) << "Verification of DAG: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of DAG: FAILURE";
      return output;
    }
  }

  // Preprocessing: values computation for necessary sorting
  ComputeBottomLevelAndWeightedLength(*tasks);
  // ComputeDepthAndTopLevel(*tasks); // already called earlier, necessary for nested conditional blocks
  if (gpto_mode == kCompComm || gpto_mode == kCompCubeComm || gpto_mode == kCompCommGroup) {
    ComputePredComm(*tasks);
  }
  if (gpto_mode == kCompCubeComm) {
    ComputePredCube(*tasks);
  }
  ComputePostOrder(*tasks);
  for (auto task : *tasks) {
    InitializeInTensorsWeight(task);
  }
  ComputeInitialMemoryImpact(*tasks);

  // Loop over all sorting combinations
  std::unordered_map<GptoTaskPtr, Time> best_start;  // to use in verify dependencies only
  std::unordered_map<GptoTaskPtr, Time> best_end;
  std::string best_solution = "";
  std::pair<std::string, Memory> best_memory_solution = std::make_pair(best_solution, MEMORY_LIMIT);
  MS_LOG(INFO) << "Start looping multiple scheduling functions";
  for (size_t task_sort = 0; task_sort < static_cast<size_t>(kNumTaskSort); ++task_sort) {
    for (size_t pes_sort = 0; pes_sort < static_cast<size_t>(PEsSort::kNumPEsSort); ++pes_sort) {
      if (SkipAlgorithm(task_sort, pes_sort)) {
        continue;
      }
      MS_LOG(INFO) << TASK_SORT_NAMES[task_sort] << " and " << PE_NAME_SORT[pes_sort];
      SchedulingOutput solution = MemAwareSchedulerCore(*tasks, type_to_num_cores_map, TASK_SORT[task_sort],
                                                        (pes_sort == static_cast<size_t>(PEsSort::kSortByLoad)));
      UpdateBestSolution(&output, solution, *tasks, &best_solution, &best_start, &best_end, task_sort, pes_sort);
      UpdateBestMemorySolution(best_solution, solution, &best_memory_solution, task_sort, pes_sort);
      for (const auto &task : *tasks) {
        task->ResetStartEnd();
      }
    }
  }
  MS_LOG(INFO) << "End looping multiple scheduling functions";

  if (best_solution == "") {
    output.makespan = SIZE_MAX;
    return output;
  }

  // Print stats about best solution
  PrintBestSolutionStats(output, *tasks, type_to_num_cores_map, best_solution, best_memory_solution);
  // Save best solution start/end values
  for (const auto &task : *tasks) {
    task->set_start(best_start[task]);
    task->set_end(best_end[task]);
  }
  return output;
}

void UpdateBestSolution(SchedulingOutput *output, const SchedulingOutput &solution,
                        const std::vector<GptoTaskPtr> &tasks, std::string *best_solution,
                        std::unordered_map<GptoTaskPtr, Time> *best_start,
                        std::unordered_map<GptoTaskPtr, Time> *best_end, size_t task_sort, size_t pe_sort) {
  if ((solution.makespan < output->makespan ||
       (solution.makespan == output->makespan && solution.memory_peak < output->memory_peak)) &&
      solution.memory_peak + PARAMETER_SIZE <= MEMORY_LIMIT) {
    *output = solution;
    *best_solution = std::string(TASK_SORT_NAMES[task_sort]) + " and " + std::string(PE_NAME_SORT[pe_sort]);
    for (const auto &task : tasks) {
      (*best_start)[task] = task->start();
      (*best_end)[task] = task->end();
    }
  }
}

void UpdateBestMemorySolution(const std::string &best_solution, const SchedulingOutput &solution,
                              std::pair<std::string, Memory> *best_memory_solution, size_t task_sort, size_t pe_sort) {
  if (solution.memory_peak < best_memory_solution->second) {
    best_memory_solution->second = solution.memory_peak;
    best_memory_solution->first =
      std::string(TASK_SORT_NAMES[task_sort]) + " and " + std::string(PE_NAME_SORT[pe_sort]);
  }
}

void PrintBestSolutionStats(const SchedulingOutput &output, const std::vector<GptoTaskPtr> &tasks,
                            const std::map<GptoTaskType, int32_t> &type_to_num_cores_map,
                            const std::string &best_solution,
                            const std::pair<std::string, Memory> &best_memory_solution) {
  const size_t kPrecision = 5;
  MS_LOG(INFO) << "Memory-aware scheduler with memory limit " << MEMORY_LIMIT;
  MS_LOG(INFO) << "Best solution is: " << best_solution;
  MS_LOG(INFO) << "Makespan of best solution is " << output.makespan;
  MS_LOG(INFO) << "Bottom level lower bound is " << LowerBoundBottomLevel(tasks);
  MS_LOG(INFO) << "Max type lower bound is " << LowerBoundPEs(tasks, type_to_num_cores_map);
  const Time total_time = TotalTime(tasks);
  MS_LOG(INFO) << "Speedup upper bound is " << std::setprecision(kPrecision)
               << 1.0 * total_time /
                    std::max(LowerBoundBottomLevel(tasks), LowerBoundPEs(tasks, type_to_num_cores_map));
  MS_LOG(INFO) << "Speedup of best solution is " << std::setprecision(kPrecision) << 1.0 * total_time / output.makespan;
  const size_t kHundred = 100;
  MS_LOG(INFO) << "Solution relative error is " << std::setprecision(kPrecision)
               << ((output.makespan /
                      (1.0 * std::max(LowerBoundBottomLevel(tasks), LowerBoundPEs(tasks, type_to_num_cores_map))) -
                    1) *
                   kHundred)
               << "%";
  MS_LOG(INFO) << "GptoTensor peak memory estimate of best solution " << output.memory_peak << " ("
               << output.memory_peak / kMBToByte << " MB)";
  MS_LOG(INFO) << "Parameter memory estimate " << PARAMETER_SIZE << " (" << PARAMETER_SIZE / kMBToByte << " MB)";
  MS_LOG(INFO) << "Total memory estimate " << output.memory_peak + PARAMETER_SIZE << " ("
               << (output.memory_peak + PARAMETER_SIZE) / kMBToByte << " MB)";
  MS_LOG(INFO) << "Best solution for memory is: " << best_memory_solution.first << " with peak memory estimate "
               << best_memory_solution.second;
}
void InitializeTasks(
  const std::vector<GptoTaskPtr> &tasks, std::unordered_map<GptoTaskId, Time> *can_start,
  std::unordered_map<GptoTaskId, size_t> *unprocessed_parents, std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks,
  std::unordered_set<GptoTaskPtr> *switch_candidates,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *left_consumers) {
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(switch_candidates);
  MS_EXCEPTION_IF_NULL(left_consumers);
  for (auto &task : tasks) {
    const auto &id = task->id();
    (*can_start)[id] = static_cast<Time>(0);
    (*unprocessed_parents)[id] = static_cast<size_t>(task->parents().size());
    if ((*unprocessed_parents)[id] == static_cast<size_t>(0)) {
      candidate_tasks->insert(task);
      if (task->condition_switch()) {
        (*switch_candidates).insert(task);
      }
    }
    for (const auto &in_tensor : task->in_tensors()) {
      (*left_consumers)[in_tensor->id()].insert(in_tensor->consumers().begin(), in_tensor->consumers().end());
    }
    task->set_minus_mem_impact(0);
  }
}

void InitializeProcessingElements(const std::map<GptoTaskType, int32_t> &type_to_num_cores_map,
                                  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *PEs_load,
                                  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *PEs_start,
                                  bool pe_load_sort) {
  MS_EXCEPTION_IF_NULL(PEs_load);
  MS_EXCEPTION_IF_NULL(PEs_start);

  size_t count = 1;
  for (const auto &type_to_num : type_to_num_cores_map) {
    const auto &type = type_to_num.first;
    const auto &num_cores = type_to_num.second;
    for (int i = 0; i < num_cores; ++i) {
      ProcessingElement new_pe;
      new_pe.id = count + i;
      new_pe.gpto_type = type;
      new_pe.load = 0;
      new_pe.idle.emplace_back(0, SIZE_MAX);
      if (pe_load_sort) {
        (*PEs_load)[type].insert(new_pe);
      } else {
        (*PEs_start)[type].push_back(new_pe);
      }
    }
    count += num_cores;
  }
}

void SubtractMemory(
  const GptoTaskPtr &selected_task,
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> *left_consumers,
  std::map<Time, Memory> *cur_mem_peak_ptr) {
  MS_EXCEPTION_IF_NULL(left_consumers);
  MS_EXCEPTION_IF_NULL(cur_mem_peak_ptr);

  auto &cur_mem_peak = *cur_mem_peak_ptr;
  for (auto &in_tensor : selected_task->in_tensors()) {
    if (selected_task->end() > in_tensor->lifetime_end()) {
      in_tensor->set_lifetime_end(selected_task->end());
      in_tensor->set_last_consumer(selected_task);
    }
    (*left_consumers)[in_tensor->id()].erase(selected_task);
    if ((*left_consumers)[in_tensor->id()].size() == 0) {
      if (in_tensor->type() == kGraphOutput) {
        continue;
      }
      for (auto it = cur_mem_peak.lower_bound(in_tensor->last_consumer().lock()->end()); it != cur_mem_peak.end();
           it++) {
        it->second -= in_tensor->weight();
      }
    }
  }

  for (auto &out_tensor : selected_task->out_tensors()) {
    out_tensor->set_lifetime_end(0);
    if (out_tensor->consumers().size() == 1 && out_tensor->type() != kGraphOutput) {
      auto last_consumer = *(out_tensor->consumers().begin());
      last_consumer.lock()->set_minus_mem_impact(last_consumer.lock()->minus_mem_impact() + out_tensor->weight());
      out_tensor->set_last_consumer(last_consumer);
    }
  }
}

void UpdateCandidates(std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks, const GptoTaskPtr &selected_task,
                      std::unordered_map<GptoTaskId, size_t> *unprocessed_parents,
                      std::unordered_map<GptoTaskId, Time> *can_start, Time *last_end, Time *last_gather_end,
                      std::unordered_set<GptoTaskPtr> *switch_candidates, const TaskSortFunction &sortPtr,
                      const size_t &position) {
  MS_EXCEPTION_IF_NULL(candidate_tasks);
  MS_EXCEPTION_IF_NULL(unprocessed_parents);
  MS_EXCEPTION_IF_NULL(can_start);
  MS_EXCEPTION_IF_NULL(last_end);
  MS_EXCEPTION_IF_NULL(last_gather_end);
  MS_EXCEPTION_IF_NULL(switch_candidates);

  // Update candidate tasks
  candidate_tasks->erase(selected_task);
  if (selected_task->condition_switch()) {
    (*switch_candidates).erase(selected_task);
  }

  // Update can_start with special processing for ConditionalSwitch/Gather cases
  if (selected_task->condition_gather()) {
    for (const auto &candidate : *candidate_tasks) {
      (*can_start)[candidate->id()] = std::max((*can_start)[candidate->id()], static_cast<Time>(selected_task->end()));
    }
    *last_gather_end = std::max(*last_gather_end, selected_task->end());
  }

  *last_end = std::max(*last_end, selected_task->end());
  for (const auto &candidate : *switch_candidates) {
    (*can_start)[candidate->id()] = std::max((*can_start)[candidate->id()], static_cast<Time>(*last_end));
  }

  // SAM update values
  const bool SAM_algo = (sortPtr >= SortBySValue && sortPtr <= SortByCostMValue);
  if (SAM_algo) {
    auto candidates = *candidate_tasks;
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
      auto updated_candidate = candidate_tasks->extract(*it);
      UPDATE_SAM[sortPtr](updated_candidate.value(), position);
      candidate_tasks->insert(std::move(updated_candidate));
    }
  }

  for (auto successor : selected_task->children()) {
    const auto &succ_id = successor->id();
    (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<Time>(selected_task->end()));
    (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<Time>(*last_gather_end));
    if (successor->condition_switch()) {
      (*can_start)[succ_id] = std::max((*can_start)[succ_id], static_cast<Time>(*last_end));
    }
    (*unprocessed_parents)[succ_id] -= 1;
    if ((*unprocessed_parents)[succ_id] == 0) {
      if (SAM_algo) {
        INIT_SAM[sortPtr](successor, position);
      }

      candidate_tasks->insert(successor);
      if (successor->condition_switch()) {
        (*switch_candidates).insert(successor);
      }
    }
  }
}

void InitializeInTensorsWeight(const GptoTaskPtr &task) {
  // can instead associate (save) list of tensors to edge when extracting scheduling input
  std::unordered_map<GptoTaskPtr, Memory> in_weight;
  Memory in_weights_sum = 0;
  for (const auto &parent : task->parents()) {
    in_weight[parent.lock()] = 0;
  }
  for (const auto &in_tensor : task->in_tensors()) {
    const auto &source = in_tensor->source();
    if (source.lock() != task) {
      in_weight[source.lock()] += in_tensor->weight();
      in_weights_sum += in_tensor->weight();
    }
  }
  task->set_in_weights(in_weight);
  task->set_in_weights_sum(in_weights_sum);
}

void InitializeS(const GptoTaskPtr &task, const size_t &current_position) {
  task->set_s_value(task->parents().size() * current_position -
                    std::accumulate(task->parents().begin(), task->parents().end(), static_cast<size_t>(0),
                                    [](size_t acc, const auto &parent) { return acc + parent.lock()->position(); }));
}

void InitializeSW(const GptoTaskPtr &task, const size_t &current_position) {
  task->set_sw_value(std::accumulate(task->parents().begin(), task->parents().end(), static_cast<Memory>(0),
                                     [current_position, task](Memory acc, const auto &parent) {
                                       return acc + (current_position - parent.lock()->position()) *
                                                      task->in_weights()[parent.lock()];
                                     }));
}

void InitializeSC(const GptoTaskPtr &task, const size_t &current_position) {
  InitializeS(task, current_position);
  task->set_sc_value(task->s_value() * task->cost());
}

void InitializeA(const GptoTaskPtr &task, const size_t &current_position) {
  InitializeS(task, current_position);
  if (task->parents().size() > 0) {
    task->set_a_value(1.0f * task->s_value() / task->parents().size());
  } else {
    task->set_a_value(0.0f);
  }
}

void InitializeAW(const GptoTaskPtr &task, const size_t &current_position) {
  InitializeSW(task, current_position);
  if (task->in_weights_sum() > 0) {
    task->set_aw_value(1.0f * task->sw_value() / task->in_weights_sum());
  } else {
    task->set_aw_value(0.0f);
  }
}

void InitializeAC(const GptoTaskPtr &task, const size_t &current_position) {
  InitializeA(task, current_position);
  task->set_ac_value(task->a_value() * task->cost());
}

void InitializeM(const GptoTaskPtr &task, const size_t &current_position) {
  size_t m = 0;
  for (const auto &parent : task->parents()) {
    m = std::max(m, current_position - parent.lock()->position());
  }
  task->set_m_value(m);
}

void InitializeMW(const GptoTaskPtr &task, const size_t &current_position) {
  Memory mw = 0;
  for (const auto &parent : task->parents()) {
    mw = std::max(
      mw, static_cast<Memory>((current_position - parent.lock()->position()) * task->in_weights()[parent.lock()]));
  }
  task->set_mw_value(mw);
}

void InitializeMC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  InitializeM(task, current_position);
  task->set_mc_value(task->m_value() * task->cost());
}

void UpdateS(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_s_value(task->s_value() + task->parents().size());
  }
}

void UpdateSW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->in_weights_sum() > 0) {
    task->set_sw_value(task->sw_value() + task->in_weights_sum());
  }
}

void UpdateSC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_sc_value(task->sc_value() + task->cost() * task->parents().size());
  }
}

void UpdateA(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_a_value(task->a_value() + 1.0);
  }
}

void UpdateAW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->in_weights_sum() > 0) {
    task->set_aw_value(task->aw_value() + 1.0);
  }
}

void UpdateAC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_ac_value(task->ac_value() + 1.0 * task->cost());
  }
}

void UpdateM(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_m_value(task->m_value() + 1);
  }
}

void UpdateMW(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  // naive for now; can improve by saving all d() values
  Memory mw = 0;
  for (const auto &parent : task->parents()) {
    mw = std::max(
      mw, static_cast<Memory>((current_position - parent.lock()->position()) * task->in_weights()[parent.lock()]));
  }
  task->set_mw_value(mw);
}

void UpdateMC(const GptoTaskPtr &task, [[maybe_unused]] const size_t &current_position = 0) {
  if (task->parents().size() > 0) {
    task->set_mc_value(task->mc_value() + task->cost());
  }
}

bool VerifyS(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      s += d;
    }
    if (s != task->s_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " s value: " << task->s_value() << " verify: " << s;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifySW(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    Memory sw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      Memory dw = d * task->in_weights()[parent.lock()];
      sw += dw;
    }
    if (sw != task->sw_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " sw value: " << task->sw_value() << " verify: " << sw;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifySC(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    Time sc = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      s += d;
    }
    sc = s * task->cost();
    if (sc != task->sc_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " sc value: " << task->sc_value() << " verify: " << sc;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyA(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    double a = 0.0;
    const double EPSILON = 1e-3;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      s += d;
    }
    if (task->parents().size() > 0) {
      a = 1.0 * s / (1.0 * task->parents().size());
    }
    if ((std::fabs(a - task->a_value()) / std::max(a, task->a_value()) > EPSILON)) {
      MS_LOG(ERROR) << "Task " << task->id() << " a value: " << task->a_value() << " verify: " << a;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyAW(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    double aw = 0.0;
    const double EPSILON = 1e-3;
    Memory sw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      Memory dw = d * task->in_weights()[parent.lock()];
      sw += dw;
    }
    if (task->in_weights_sum() > 0) {
      aw = 1.0 * sw / (1.0 * task->in_weights_sum());
    }
    if ((std::fabs(aw - task->aw_value()) / std::max(aw, task->aw_value()) > EPSILON)) {
      MS_LOG(ERROR) << "Task " << task->id() << " aw value: " << task->aw_value() << " verify: " << aw;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyAC(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t s = 0;
    double ac = 0.0;
    const double EPSILON = 1e-3;
    for (const auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      s += d;
    }
    if (task->parents().size() > 0) {
      const double a = 1.0 * s / (1.0 * task->parents().size());
      ac = a * task->cost();
    }
    if ((std::fabs(ac - task->ac_value()) > EPSILON) / std::max(ac, task->ac_value()) > EPSILON) {
      MS_LOG(ERROR) << "Task " << task->id() << " ac value: " << task->ac_value() << " verify: " << ac;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyM(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t m = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      if (d > m) {
        m = d;
      }
    }
    if (m != task->m_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " m value: " << task->m_value() << " verify: " << m;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyMW(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    Memory mw = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      Memory dw = d * task->in_weights()[parent.lock()];
      if (dw > mw) {
        mw = dw;
      }
    }
    if (mw != task->mw_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " mw value: " << task->mw_value() << " verify: " << mw;
      success = false;
      break;
    }
  }
  return success;
}

bool VerifyMC(const std::vector<GptoTaskPtr> &tasks) {
  bool success = true;
  for (const auto &task : tasks) {
    size_t m = 0;
    Time mc = 0;
    for (auto &parent : task->parents()) {
      size_t d = task->position() - parent.lock()->position();
      if (d > m) {
        m = d;
      }
    }
    mc = m * task->cost();
    if (mc != task->mc_value()) {
      MS_LOG(ERROR) << "Task " << task->id() << " mc value: " << task->mc_value() << " verify: " << mc;
      success = false;
      break;
    }
  }
  return success;
}

SchedulingOutput MemAwareSchedulerCore(const std::vector<GptoTaskPtr> &tasks,
                                       const std::map<GptoTaskType, int32_t> &type_to_num_cores_map,
                                       const TaskSortFunction &sortPtr, bool pe_load_sort) {
  SchedulingOutput output{{}, 0, 0};
  output.task_times.reserve(tasks.size());

  // Initializations for tasks
  std::set<GptoTaskPtr, TaskSortFunction> candidate_tasks(sortPtr);
  std::unordered_map<GptoTaskId, Time> can_start;
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::unordered_set<GptoTaskPtr> switch_candidates;
  std::unordered_map<size_t, std::set<std::weak_ptr<GptoTask>, GptoTask::SortByIdWeak>> left_consumers;
  InitializeTasks(tasks, &can_start, &unprocessed_parents, &candidate_tasks, &switch_candidates, &left_consumers);

  // Initializations for processing elements
  // Pick a sorting for processing elements
  // Implemented: SortByLoad, SortByAvailableStart
  // Only one structure to be used depending on argument; we define both here
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> PEs_load;
  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> PEs_start;
  InitializeProcessingElements(type_to_num_cores_map, &PEs_load, &PEs_start, pe_load_sort);

  // Task graph scheduling loop
  std::map<Time, Memory> cur_mem_peak;
  Time last_end = 0;
  Time last_gather_end = 0;
  size_t position = 0;
  size_t last_switch_subgraph = SIZE_MAX;
  while (!candidate_tasks.empty()) {
    // Schedule a task (if possible)
    std::tuple<GptoTaskPtr, Time, PeId> scheduled_info;
    if (pe_load_sort) {
      scheduled_info = ScheduleTaskLoad(&candidate_tasks, &PEs_load, &can_start, &cur_mem_peak, &last_switch_subgraph);
    } else {
      scheduled_info =
        ScheduleTaskStart(&candidate_tasks, &PEs_start, &can_start, &cur_mem_peak, &last_switch_subgraph);
    }

    GptoTaskPtr &selected_task = std::get<0>(scheduled_info);
    Time &selected_time = std::get<1>(scheduled_info);
    PeId &selected_pe = std::get<2>(scheduled_info);

    if (selected_task == nullptr) {  // Out-of-memory for all candidates
      output.makespan = SIZE_MAX;
      output.memory_peak = SIZE_MAX;
      MS_LOG(INFO) << "Out of memory estimated!";
      return output;
    }

    // Switch/gather logic
    if (selected_task->condition_switch()) {
      last_switch_subgraph = selected_task->subgraph_id();
    } else if (selected_task->condition_gather()) {
      last_switch_subgraph = selected_task->subgraph_id_parent();
    }

    // Update output
    selected_task->set_position(position++);
    selected_task->set_start(selected_time);
    selected_task->set_end(selected_time + selected_task->cost());
    Interval new_interval{selected_task, selected_task->start(), selected_task->end(), selected_pe};
    output.task_times.push_back(new_interval);
    output.makespan = std::max(output.makespan, selected_task->end());

    // Bookkeeping for memory consumption and candidates
    AddMemory(selected_task, selected_time, &cur_mem_peak);
    SubtractMemory(selected_task, &left_consumers, &cur_mem_peak);
    UpdateCandidates(&candidate_tasks, selected_task, &unprocessed_parents, &can_start, &last_end, &last_gather_end,
                     &switch_candidates, sortPtr, position);
  }  // end-while candidates not empty

  // End of time: reset memory to zero (equivalent to subtracting kGraphOutput tensors)
  cur_mem_peak.rbegin()->second = 0;

  // Output memory peak is the max of cur_mem_peak
  output.memory_peak = 0;
  for (const auto &time_mem : cur_mem_peak) {
    if (time_mem.second > output.memory_peak) {
      output.memory_peak = time_mem.second;
    }
  }
  // Print result
  MS_LOG(INFO) << "Makespan is " << output.makespan;
  MS_LOG(INFO) << "Peak mem is " << output.memory_peak;

  // Verification of scheduling solution (optional)
  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    if (VerifyScheduling(tasks)) {
      MS_LOG(INFO) << "Verification of Scheduling: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of Scheduling: FAILURE";
      output.makespan = SIZE_MAX;
    }

    if (VerifyMemory(tasks, &cur_mem_peak)) {
      MS_LOG(INFO) << "Verification of Memory: SUCCESS";
    } else {
      MS_LOG(ERROR) << "Verification of Memory: FAILURE";
    }

    if (sortPtr >= SortBySValue && sortPtr <= SortByCostMValue) {
      if (VERIFY_SAM[sortPtr](tasks)) {
        MS_LOG(INFO) << "Verification of SAM: SUCCESS";
      } else {
        MS_LOG(ERROR) << "Verification of SAM: FAILURE";
      }
    }
  }

  return output;
}

std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskLoad(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks_ptr,
  std::unordered_map<GptoTaskType, std::set<ProcessingElement, SortByLoad>> *PEs_load_ptr,
  std::unordered_map<GptoTaskId, Time> *can_start, std::map<Time, Memory> *cur_mem_peak_ptr,
  size_t *last_switch_subgraph_ptr) {
  auto &candidate_tasks = *candidate_tasks_ptr;
  auto &PEs_load = *PEs_load_ptr;
  auto &cur_mem_peak = *cur_mem_peak_ptr;
  auto &last_switch_subgraph = *last_switch_subgraph_ptr;

  bool exists_subgraph = false;
  for (auto task_it = candidate_tasks.begin(); task_it != candidate_tasks.end(); ++task_it) {
    auto &selected_task = *task_it;
    const auto &selected_id = selected_task->id();
    auto &PEs = PEs_load[selected_task->gpto_type()];
    // Pick a PE amongst PEs of same type
    for (auto pe_it = PEs.begin(); pe_it != PEs.end(); pe_it++) {
      auto &mut_pe = const_cast<ProcessingElement &>(*pe_it);
      // Put in first idle window that fits the task (if memory limit is not violated)
      for (auto idle_it = mut_pe.idle.begin(); idle_it != mut_pe.idle.end(); ++idle_it) {
        Time start_time;
        bool case_flag = false;
        // Distinguish cases based on can_start constraint
        if ((*can_start)[selected_id] <= idle_it->first) {
          start_time = idle_it->first;
        } else if ((*can_start)[selected_id] <= idle_it->second) {
          start_time = (*can_start)[selected_id];
          case_flag = true;
        } else {  // (*can_start)[selected_id] > idle_it->second (not allowed to place here)
          continue;
        }
        if (idle_it->second - start_time < selected_task->cost() ||
            MemoryViolated(selected_task, start_time, &cur_mem_peak, &last_switch_subgraph, &exists_subgraph)) {
          continue;
        }
        // Place task in idle window here
        // Save info to return: start task at time idle_it->first
        PeId selected_pe = (*pe_it).id;
        Time selected_time = start_time;
        // Update idle list
        if (!case_flag) {
          if (idle_it->second - idle_it->first ==
              selected_task->cost()) {  // whole idle interval is filled in, erase it
            mut_pe.idle.erase(idle_it);
          } else {  // idle_it->second - idle_it->first > selected_task->cost()
            idle_it->first += selected_task->cost();
          }
        } else {  // case_flag = true, idle interval is broken into two
                  // sub-blocks [idle_it->first, can_start] and
                  // (maybe empty) [can_start + cost, idle_it->second]
          Time upper = idle_it->second;
          idle_it->second = (*can_start)[selected_id];
          if (upper - (*can_start)[selected_id] - selected_task->cost() > 0) {
            std::pair<Time, Time> new_idle = std::make_pair((*can_start)[selected_id] + selected_task->cost(), upper);
            mut_pe.idle.emplace(std::next(idle_it), new_idle);
          }
        }
        // Update load, PEs, and memory peaks
        auto updated_PE = PEs.extract(pe_it);
        updated_PE.value().load += selected_task->cost();
        PEs.insert(std::move(updated_PE));
        return std::make_tuple(selected_task, selected_time, selected_pe);
      }  // end-for idle windows
    }
  }
  return std::make_tuple(nullptr, 0, 0);
}

std::tuple<GptoTaskPtr, Time, PeId> ScheduleTaskStart(
  std::set<GptoTaskPtr, TaskSortFunction> *candidate_tasks_ptr,
  std::unordered_map<GptoTaskType, std::vector<ProcessingElement>> *PEs_start_ptr,
  std::unordered_map<GptoTaskId, Time> *can_start, std::map<Time, Memory> *cur_mem_peak_ptr,
  size_t *last_switch_subgraph_ptr) {
  auto &candidate_tasks = *candidate_tasks_ptr;
  auto &PEs_start = *PEs_start_ptr;
  auto &cur_mem_peak = *cur_mem_peak_ptr;
  auto &last_switch_subgraph = *last_switch_subgraph_ptr;

  bool exists_subgraph = false;
  for (auto task_it = candidate_tasks.begin(); task_it != candidate_tasks.end(); ++task_it) {
    auto &selected_task = *task_it;
    const auto &selected_id = selected_task->id();
    auto &PEs = PEs_start[selected_task->gpto_type()];
    // Precompute min first available start for task
    Time min_start = SIZE_MAX;
    bool min_case = false;
    std::vector<ProcessingElement>::iterator min_it;
    std::list<std::pair<Time, Time>>::iterator min_idle_it;
    for (auto it = PEs.begin(); it != PEs.end(); ++it) {
      for (auto idle_it = it->idle.begin(); idle_it != it->idle.end(); ++idle_it) {
        Time start_time;
        bool case_flag = false;
        // Distinguish cases based on can_start constraint
        if ((*can_start)[selected_id] <= idle_it->first) {
          start_time = idle_it->first;
        } else if ((*can_start)[selected_id] <= idle_it->second) {
          start_time = (*can_start)[selected_id];
          case_flag = true;
        } else {  // (*can_start)[selected_id] > idle_it->second (not allowed to place here)
          continue;
        }
        if (idle_it->second - start_time >= selected_task->cost()) {
          if (MemoryViolated(selected_task, start_time, &cur_mem_peak, &last_switch_subgraph, &exists_subgraph)) {
            continue;
          }
          if (min_start > start_time) {
            min_start = start_time;
            min_case = case_flag;
            min_it = it;
            min_idle_it = idle_it;
            break;
          }
        }
      }
    }
    if (min_start == SIZE_MAX) {  // cannot place selected_task anywhere
      if (std::next(task_it) == candidate_tasks.end()) {
        return std::make_tuple(nullptr, 0, 0);
      } else {
        continue;
      }
    }
    // Assign task to min-start-time PE
    auto selected_pe = (*min_it).id;
    auto selected_time = min_start;
    // Update idle list
    if (!min_case) {
      if (min_idle_it->second - min_idle_it->first ==
          selected_task->cost()) {  // whole idle interval is filled in, erase it
        min_it->idle.erase(min_idle_it);
      } else {  // idle_it->second - idle_it->first > selected_task->cost()
        min_idle_it->first += selected_task->cost();
      }
    } else {  // min_case = true, idle interval is broken into two sub-blocks
              // [idle_it->first, can_start] and
              // (maybe empty) [can_start + task.cost(), idle_it->second]
      Time upper = min_idle_it->second;
      min_idle_it->second = (*can_start)[selected_id];
      if (upper - (*can_start)[selected_id] - selected_task->cost() > 0) {
        std::pair<Time, Time> new_idle = std::make_pair((*can_start)[selected_id] + selected_task->cost(), upper);
        min_it->idle.emplace(std::next(min_idle_it), new_idle);
      }
    }
    // Update and break
    min_it->load += selected_task->cost();
    return std::make_tuple(selected_task, selected_time, selected_pe);
  }
  return std::make_tuple(nullptr, 0, 0);
}

void AddMemory(const GptoTaskPtr &selected_task, const Time &selected_time, std::map<Time, Memory> *cur_mem_peak_ptr) {
  auto &cur_mem_peak = *cur_mem_peak_ptr;
  bool found_start = false;
  bool found_end = false;
  Time prev_end = 0;
  auto selected_time_it = cur_mem_peak.lower_bound(selected_time);
  for (auto it = selected_time_it; it != cur_mem_peak.end(); it++) {
    auto &time = it->first;
    auto &mem = it->second;
    if (time == selected_time) {
      found_start = true;
      prev_end = std::max(prev_end, time);
      mem += selected_task->initial_mem_impact();
    } else if (time < selected_time + selected_task->cost()) {
      prev_end = std::max(prev_end, time);
      mem += selected_task->initial_mem_impact();
    } else if (time == selected_time + selected_task->cost()) {
      found_end = true;
      mem += selected_task->initial_mem_impact() - selected_task->workspace_memory();
    } else if (time > selected_time + selected_task->cost()) {
      mem += selected_task->initial_mem_impact() - selected_task->workspace_memory();
    }
  }
  if (!found_start) {
    if (selected_time_it != cur_mem_peak.begin() && selected_time_it != cur_mem_peak.end()) {
      cur_mem_peak[selected_time] = std::prev(selected_time_it)->second + selected_task->initial_mem_impact();
    } else {
      cur_mem_peak[selected_time] = selected_task->initial_mem_impact();
    }
  }
  if (!found_end) {
    cur_mem_peak[selected_time + selected_task->cost()] =
      cur_mem_peak[std::max(prev_end, selected_time)] - selected_task->workspace_memory();
  }
}

bool MemoryViolated(const GptoTaskPtr &selected_task, const Time &start_time, std::map<Time, Memory> *cur_mem_peak_ptr,
                    size_t *last_switch_subgraph, bool *exists_subgraph) {
  if (selected_task->subgraph_id() < *last_switch_subgraph) {
    *exists_subgraph = true;
    return MemoryViolatedCore(selected_task, start_time, cur_mem_peak_ptr);
  } else {  // selected_task->subgraph_id() == *last_switch_subgraph
    if (*exists_subgraph) {
      return false;
    } else {
      return MemoryViolatedCore(selected_task, start_time, cur_mem_peak_ptr);
    }
  }
}

bool MemoryViolatedCore(const GptoTaskPtr &selected_task, const Time &start_time,
                        std::map<Time, Memory> *cur_mem_peak_ptr) {
  auto &cur_mem_peak = *cur_mem_peak_ptr;
  auto start_time_it = cur_mem_peak.lower_bound(start_time);
  for (auto it = start_time_it; it != cur_mem_peak.end(); it++) {
    auto &time = it->first;
    auto &mem = it->second;
    if (time < start_time + selected_task->cost()) {
      if (PARAMETER_SIZE + mem + selected_task->initial_mem_impact() > MEMORY_LIMIT) {
        return true;
      }
    } else {  // time >= start_time + selected_task->cost()
      if (PARAMETER_SIZE + mem + selected_task->initial_mem_impact() - selected_task->minus_mem_impact() -
            selected_task->workspace_memory() >
          MEMORY_LIMIT) {
        return true;
      }
    }
  }
  if (cur_mem_peak.find(start_time) == cur_mem_peak.end()) {
    if (start_time_it != cur_mem_peak.begin() && start_time_it != cur_mem_peak.end()) {
      if (PARAMETER_SIZE + std::prev(start_time_it)->second + selected_task->initial_mem_impact() > MEMORY_LIMIT) {
        return true;
      }
    }
  }
  if (cur_mem_peak.find(start_time + selected_task->cost()) == cur_mem_peak.end()) {
    auto end_time_it = cur_mem_peak.lower_bound(start_time + selected_task->cost());
    if (end_time_it != cur_mem_peak.begin() && end_time_it != cur_mem_peak.end()) {
      if (PARAMETER_SIZE + std::prev(end_time_it)->second + selected_task->initial_mem_impact() -
            selected_task->minus_mem_impact() - selected_task->workspace_memory() >
          MEMORY_LIMIT) {
        return true;
      }
    }
  }
  return false;
}

bool VerifyScheduling(const std::vector<GptoTaskPtr> &tasks) {
  bool flag = true;
  MS_LOG(INFO) << "Start Verification of Scheduling";
  for (const auto &task : tasks) {
    // Check if task is scheduled before its children
    for (auto child = task->children().begin(); child != task->children().end(); ++child) {
      if (!(task->start() < task->end() && task->end() <= (*child)->start() &&
            (*child)->start() < (*child)->end())) {  // assume open-rightpoint intervals and non-zero size
        MS_LOG(INFO) << "Verification violation: task " << task->id() << " [" << task->start() << "," << task->end()
                     << "] and task " << (*child)->id() << " [" << (*child)->start() << "," << (*child)->end() << "]";
        flag = false;
      }
    }
  }
  MS_LOG(INFO) << "End Verification of Scheduling";
  return flag;
}

bool VerifyDAG(const std::vector<GptoTaskPtr> &tasks) {
  // simple verifier: no directed cycle exists
  std::unordered_map<GptoTaskId, bool> visited;
  std::unordered_map<GptoTaskId, size_t> unprocessed_parents;
  std::deque<GptoTaskPtr> to_visit;
  MS_LOG(INFO) << "Start Verification of DAG";
  for (const auto &task : tasks) {
    const auto &id = task->id();
    visited[id] = false;
    unprocessed_parents[id] = task->parents().size();
    if (unprocessed_parents[id] == 0) {
      to_visit.push_back(task);
    }
  }
  while (!to_visit.empty()) {
    const auto selected_task = *(to_visit.begin());
    const auto &selected_id = selected_task->id();
    if (visited[selected_id]) {
      MS_LOG(INFO) << "Cycle including task " << selected_id;
      return false;
    } else {
      visited[selected_id] = true;
    }
    to_visit.pop_front();
    for (const auto &successor : selected_task->children()) {
      const auto &succ_id = successor->id();
      unprocessed_parents[succ_id] -= 1;
      if (unprocessed_parents[succ_id] == 0) {
        to_visit.push_back(successor);
      }
    }
  }
  MS_LOG(INFO) << "End Verification of DAG";
  return true;
}

bool VerifyMemory(const std::vector<GptoTaskPtr> &tasks, std::map<Time, Memory> *mem_peak) {
  bool verified = true;
  std::map<Time, Memory> verify_peak;
  Memory graph_output_mem = 0;

  for (auto &time_mem : *mem_peak) {
    verify_peak[time_mem.first] = 0;
  }

  // Recompute peaks based on tensor lifetimes
  for (auto &task : tasks) {
    std::vector<GptoTensorPtr> tensors;
    tensors.reserve(task->out_tensors().size() + task->workspace_tensors().size());
    std::copy(task->out_tensors().begin(), task->out_tensors().end(), std::back_inserter(tensors));
    std::copy(task->workspace_tensors().begin(), task->workspace_tensors().end(), std::back_inserter(tensors));
    for (auto &tensor : tensors) {
      const auto &start = tensor->source().lock()->start();
      Time end;
      if (tensor->consumers().size() > 0) {
        end = tensor->lifetime_end();
      } else {
        end = tensor->source().lock()->end();
      }
      if (tensor->type() == kGraphOutput) {
        graph_output_mem += tensor->weight();
      }
      for (auto &time_mem : *mem_peak) {
        if ((start <= time_mem.first && time_mem.first < end && tensor->type() != kGraphOutput) ||
            (start <= time_mem.first && tensor->type() == kGraphOutput)) {
          verify_peak[time_mem.first] += tensor->weight();
        }
      }
    }
  }

  if (verify_peak.rbegin()->second != graph_output_mem) {
    MS_LOG(ERROR) << "Time " << verify_peak.rbegin()->first << " verify peak memory " << verify_peak.rbegin()->second
                  << " kGraphOutput memory " << graph_output_mem;
    verified = false;
  }
  verify_peak.rbegin()->second = 0;

  // Compare to originally saved peaks
  for (auto &time_mem : *mem_peak) {
    if ((*mem_peak)[time_mem.first] != verify_peak[time_mem.first]) {
      MS_LOG(ERROR) << "Time " << time_mem.first << " peak " << (*mem_peak)[time_mem.first] << " verify peak "
                    << verify_peak[time_mem.first];
      verified = false;
    }
  }
  return verified;
}

void LogSchedulingOutput(const SchedulingInput &scheduling_input, const SchedulingOutput &output,
                         const std::unordered_map<CNodePtr, GptoTaskPtr> &cnode_to_task,
                         const KernelGraphPtr &kernel_graph, const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors,
                         const Memory memory_lower_bound, [[maybe_unused]] std::map<std::string, size_t> *group_to_id,
                         const std::string &path) {
  auto lower_makespan =
    std::max(LowerBoundBottomLevel(scheduling_input.tasks), LowerBoundPEs(scheduling_input.tasks, GetPEs(group_to_id)));
  const size_t graph_id = kernel_graph->graph_id();
  std::stringstream ss;
  std::ostringstream oss;
  std::string filename;
  ss << kernel_graph;
  filename = path + std::string("/") + std::string("gpto_out_") + std::to_string(graph_id) + std::string("_") +
             ss.str() + std::string(".log");
  // Print info for tasks
  const auto &intervals = output.task_times;
  for (const auto &interval : intervals) {
    oss << "TASK id=" << std::to_string(interval.task->id()) << ", name=" << interval.task->name()
        << ", type=" << std::to_string(interval.task->gpto_type()) << ", cost=" << std::to_string(interval.task->cost())
        << ", start=" << std::to_string(interval.start) << ", end=" << std::to_string(interval.end)
        << ", pe=" << std::to_string(interval.pid) << ", subgraph=" << std::to_string(interval.task->subgraph_id())
        << ", subgraph_parent=" << std::to_string(interval.task->subgraph_id_parent()) << "\n";
  }
  // Print events
  for (const auto &interval : intervals) {
    const auto &dst_id = interval.task->id();
    for (const auto &source : interval.task->recv_events()) {
      oss << "EVENT " << std::to_string(source.lock()->id()) << " " << std::to_string(dst_id) << "\n";
    }
  }

  // Print makespan and memory bounds
  oss << "UPPER " << output.makespan << "\n";
  oss << "LOWER " << lower_makespan << "\n";
  oss << "MEMORY_LIMIT " << MEMORY_LIMIT << "\n";
  oss << "PARAMETER_SIZE " << PARAMETER_SIZE << "\n";
  oss << "MEMORY LOWER BOUND " << memory_lower_bound << "\n";

  // Print edges
  for (const auto &interval : intervals) {
    for (const auto &child : interval.task->children()) {
      oss << "EDGE " << std::to_string(interval.task->id()) << " " << std::to_string(child->id()) << "\n";
    }
  }

  // Print tensor info
  for (const auto &tensor : tensors) {
    std::string consumers = std::accumulate(tensor->consumers().begin(), tensor->consumers().end(), std::string{},
                                            [](std::string consumers_str, const auto &consumer) {
                                              return consumers_str += std::to_string(consumer.lock()->id()) + ";";
                                            });
    oss << "TENSOR id=" << std::to_string(tensor->id()) << ", weight=" << std::to_string(tensor->weight())
        << ", source=" << std::to_string(tensor->source().lock()->id()) << ", type=" << std::to_string(tensor->type())
        << ", consumers=" << consumers << "\n";
  }

  (void)Common::SaveStringToFile(filename, oss.str());
}

void ComputeAncestorsDescendants(const std::vector<GptoTaskPtr> &tasks,
                                 std::vector<mindspore::somas::DynamicBitSet> *nodes_dependency) {
  // Assume tasks are sorted by id (ie in BFS order); if not, sort them
  // Do we need each node to be ancestor/descendant of itself? No (for now)

  MS_EXCEPTION_IF_NULL(nodes_dependency);
  MS_EXCEPTION_IF_NULL(tasks.back());
  size_t count = tasks.back()->id() + 1;
  for (size_t i = 0; i < count; i++) {
    (void)nodes_dependency->emplace_back(count);
  }
  for (const auto &task : tasks) {
    for (const auto &parent : task->parents()) {
      auto &elem = (*nodes_dependency)[task->id()];
      elem.SetBitTrue(parent.lock()->id());
      Union(&((*nodes_dependency)[task->id()]), &((*nodes_dependency)[parent.lock()->id()]));
    }
    // Log message just for debugging
    MS_LOG(DEBUG) << "Task " << task->id() << " has " << (*nodes_dependency)[task->id()].CountOnesNum() << "ancestors";
  }
}

void InsertEdges(const KernelGraphPtr &kernel_graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  const std::list<CNodePtr> &cnode_list = kernel_graph->GetOrderedCnodes();
  std::vector<CNodePtr> cnode_vec(cnode_list.cbegin(), cnode_list.cend());
  auto &cnode_to_task = *cnode_to_task_map_ptr;

  std::queue<CNodePtr> visit_queue;
  std::unordered_map<CNodePtr, size_t> unprocessed_children;
  std::unordered_map<CNodePtr, std::unordered_set<GptoTaskPtr>> real_children;

  // Initialization loops
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    unprocessed_children[cnode_vec[i]] = 0;
  }
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    for (size_t j = 0; j < cnode_vec[i]->size(); ++j) {
      if (!(cnode_vec[i]->input(j)->isa<CNode>())) {
        continue;
      }
      const auto &input_node = cnode_vec[i]->input(j)->cast<CNodePtr>();
      unprocessed_children[input_node] = unprocessed_children[input_node] + 1;
    }
  }
  for (size_t i = 0; i < cnode_vec.size(); ++i) {
    if (unprocessed_children[cnode_vec[i]] == 0) {
      visit_queue.push(cnode_vec[i]);
    }
  }

  // CNode graph traversal loop
  while (!visit_queue.empty()) {
    const auto &visit_cnode = visit_queue.front();
    MS_LOG(DEBUG) << "Visit cnode " << visit_cnode->UniqueName();
    if (cnode_to_task.count(visit_cnode) > 0) {  // if real kernel, then add edges
      const auto &visit_task = cnode_to_task[visit_cnode];
      for (const auto &real_child : real_children[visit_cnode]) {
        visit_task->AddChild(real_child);
        real_child->AddParent(visit_task);
        MS_LOG(DEBUG) << "Edge " << visit_task->id() << " " << real_child->id();
        MS_LOG(DEBUG) << "Edge (UniqueName) " << visit_cnode->UniqueName() << " " << real_child->cnode()->UniqueName();
      }
      real_children[visit_cnode].clear();
      real_children[visit_cnode].insert(visit_task);
    }
    // Maintain real_children and visit_queue
    for (size_t j = 1; j < visit_cnode->size(); ++j) {
      if (!visit_cnode->input(j)->isa<CNode>()) {
        continue;
      }
      const auto &parent_cnode = visit_cnode->input(j)->cast<CNodePtr>();
      for (const auto &real_child : real_children[visit_cnode]) {
        real_children[parent_cnode].insert(real_child);
      }
      unprocessed_children[parent_cnode]--;
      if (unprocessed_children[parent_cnode] == 0) {
        visit_queue.push(parent_cnode);
      }
    }
    visit_queue.pop();
  }
}

bool IsCubeKernel(const CNodePtr &node) {
  static const std::unordered_set<std::string> kCubeKernelSet = {
    // matmul
    kMatMulOpName, kMatMulV2OpName, kBatchMatMulOpName, kBatchMatMulV2OpName,
    // conv
    kConv2DOpName, kConv3DOpName,
    // conv dx
    kConv2DBackpropInputOpName, kConv2DBackpropInputDOpName, kConv2DTransposeOpName, kConv2DTransposeDOpName,
    kDepthwiseConv2DBackpropInputOpName, kDepthwiseConv2DBackpropInputDOpName, kConv3DBackpropInputOpName,
    kConv3DBackpropInputDOpName, kConv3DTransposeOpName, kConv3DTransposeDOpName,
    // conv dw
    kConv2DBackpropFilterOpName, kConv2DBackpropFilterDOpName, kDepthwiseConv2DBackpropFilterOpName,
    kDepthwiseConv2DBackpropFilterDOpName, kConv3DBackpropFilterOpName, kConv3DBackpropFilterDOpName};

  auto op_name = common::AnfAlgo::GetCNodeName(node);
  return kCubeKernelSet.find(op_name) != kCubeKernelSet.end();
}

GptoTaskType GetType(mindspore::device::DeviceResManager *device_res_manager, const CNodePtr cnode,
                     [[maybe_unused]] std::map<std::string, size_t> *group_to_id) {
  if (gpto_mode == kComp) {
    return kVec;
  } else if (gpto_mode == kCompComm) {
    return common::AnfAlgo::IsCommunicationOp(cnode) ? kComm : kVec;
  } else if (gpto_mode == kCompCubeComm) {
    if (common::AnfAlgo::IsCommunicationOp(cnode)) {
      return kComm;
    } else {
      return IsCubeKernel(cnode) ? kCube : kVec;
    }
  } else {  // kCompCommGroup
    if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
      return kVec;
    } else {
      const auto prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      const auto group_value = prim->GetAttr(kAttrGroup);
      if (group_value == nullptr) {
        MS_LOG(EXCEPTION) << "Group value is nullptr, node: " << cnode->fullname_with_scope();
      }
      if (group_value->isa<StringImm>()) {
        const auto &group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
        size_t comm_group = device_res_manager->GetCommunicationStreamIDByGroup(group_name);
        if (group_to_id->find(group_name) == group_to_id->end()) {
          (*group_to_id)[group_name] = comm_group;
        }
        return comm_group;
      } else {
        return kVec;
      }
    }
  }
}

GptoTaskType GetRealType(const CNodePtr cnode) {
  if (common::AnfAlgo::IsCommunicationOp(cnode)) {
    return kComm;
  } else if (IsCubeKernel(cnode)) {
    return kCube;
  } else {
    return kComp;
  }
}

size_t GetAlignedSize(size_t original_size) {
  constexpr size_t alignment = 512;
  constexpr size_t alignment_complement = 31;
  size_t aligned_size =
    (original_size > 0) ? ((original_size + alignment + alignment_complement) / alignment) * alignment : 0;
  return aligned_size;
}

KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index,
                                             std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  auto prenode = common::AnfAlgo::VisitKernelWithReturnType(ori_node, ori_index, false);
  while (prenode.first->isa<CNode>() &&
         cnode_to_task_map_ptr->find(prenode.first->cast<CNodePtr>()) == cnode_to_task_map_ptr->end()) {
    auto cnode = prenode.first->cast<CNodePtr>();
    if (!common::AnfAlgo::IsNopNode(cnode) &&
        !(IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && common::AnfAlgo::GetInputNum(cnode) == 1)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Node[" << ori_node->fullname_with_scope() << "] find input node["
                                 << cnode->fullname_with_scope()
                                 << "] doesn't exist in cnode_to_task map and is not a nop node!";
    }
    prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
  }
  return prenode;
}

void ExtractOutputWorkspaceTensors(const SchedulingInput &scheduling_input, const std::vector<GptoTaskPtr> &tasks) {
  size_t tensor_count = 0;
  // Looping over tasks to obtain output and workspace tensors (somas style)
  for (auto &task : tasks) {
    const auto &kernel_mod = AnfAlgo::GetKernelMod(task->cnode());
    MS_EXCEPTION_IF_NULL(kernel_mod);

    // Extract each node's output tensors
    task->out_tensors().reserve(kernel_mod->GetOutputSizeList().size());
    for (const auto &size : kernel_mod->GetOutputSizeList()) {
      Memory weight = GetAlignedSize(size);
      if (weight == 0) {
        weight = GetAlignedSize(1);
      }
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task,
                                                              kWorkspace);  // initially kWorkspace, since no consumers
      task->out_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New output tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
      tensor_count++;
    }

    // Extract each node's workspace tensor
    task->workspace_tensors().reserve(kernel_mod->GetWorkspaceSizeList().size());
    for (const auto &size : kernel_mod->GetWorkspaceSizeList()) {
      Memory weight = GetAlignedSize(size);
      if (weight == 0) {
        weight = GetAlignedSize(1);
      }
      GptoTensorPtr new_tensor = std::make_shared<GptoTensor>(tensor_count, size, weight, task, kWorkspace);
      task->workspace_tensors().push_back(new_tensor);
      MS_LOG(DEBUG) << "New workspace tensor " << tensor_count << " source id " << task->id() << " weight " << weight;
      tensor_count++;
    }
  }
  MAX_TENSOR_ID = tensor_count - 1;
}

void CleanWorkspace(CNodePtr pre_node, const GptoTaskPtr &pre_task, const GptoTaskPtr &task) {
  auto clean_workspace_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
  for (const auto &index : clean_workspace_indexs) {
    if (index > pre_task->out_tensors().size()) {
      MS_LOG(INFO) << "Workspace index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                   << "]'s Workspace size " << pre_task->workspace_tensors().size();
      continue;
    }
    auto input_tensor = pre_task->workspace_tensors()[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    task->in_tensors().insert(input_tensor);
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
    input_tensor->consumers().insert(task);
  }
}

void CleanOutput(size_t index, CNodePtr pre_node, GptoTaskPtr pre_task, const GptoTaskPtr &task) {
  if (index > pre_task->out_tensors().size()) {
    MS_LOG(INFO) << "Output index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                 << "]'s outputs size " << pre_task->out_tensors().size();
    return;
  }
  auto input_tensor = pre_task->out_tensors()[index];
  MS_EXCEPTION_IF_NULL(input_tensor);
  task->in_tensors().insert(input_tensor);
  if (input_tensor->type() == GptoTensorType::kWorkspace) {
    input_tensor->set_type(GptoTensorType::kSimple);
  }
  input_tensor->consumers().insert(task);
}

void StandardInputCase(const GptoTaskPtr &task, std::unordered_set<void *> *parameter_set, const CNodePtr &kernel,
                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr) {
  MS_EXCEPTION_IF_NULL(parameter_set);

  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);

  const auto &input_size_list = AnfAlgo::GetNodeInputSizeList(kernel);
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  MS_LOG(DEBUG) << "StandardInputCase Task " << task->id() << " " << task->cnode()->fullname_with_scope();
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    KernelWithIndex prenode_index = GetVisitKernelWithReturnType(input_node, 0, cnode_to_task_gpto_map_ptr);
    MS_EXCEPTION_IF_NULL(prenode_index.first);
    if (common::AnfAlgo::CheckPrimitiveType(prenode_index.first, prim::kPrimMakeTuple)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Node " << kernel->fullname_with_scope() << "'s input node ["
                                 << input_node->DebugString() << "]'s input " << i << " is MakeTuple";
    }

    if (!AnfUtils::IsRealCNodeKernel(prenode_index.first)) {  // somas input parameter case
      MS_LOG(DEBUG) << "Input  [" << prenode_index.first->fullname_with_scope() << "] is not a real cnode kernel.";
      MS_LOG(DEBUG) << "Checking input parameter";
      auto op_name = common::AnfAlgo::GetCNodeName(kernel);
      TypeId input_origin_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel, i);
      if ((op_name == kDynamicRNNOpName || op_name == kDynamicGRUV2OpName) && input_origin_type == kMetaTypeNone) {
        continue;
      }
      size_t input_size = 0;
      if (i >= input_size_list.size()) {
        MS_LOG(DEBUG) << "Node: " << kernel->fullname_with_scope() << " input idx: " << i
                      << " greater than the size of input_size_list: " << input_size_list.size()
                      << ", so use 0 as parameter size.";
      } else {
        input_size = input_size_list.at(i);
      }
      if (parameter_set->find(prenode_index.first.get()) == parameter_set->end()) {
        parameter_set->insert(prenode_index.first.get());
        PARAMETER_SIZE += input_size;
      }
      continue;
    }
    auto iter = cnode_to_task_gpto_map_ptr->find(prenode_index.first->cast<CNodePtr>());
    if (iter == cnode_to_task_gpto_map_ptr->end()) {
      MS_LOG(DEBUG) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                    << prenode_index.first->fullname_with_scope() << "] not found in tasks";
      continue;
    }
    auto pre_task = iter->second;
    if (pre_task->out_tensors().size() == 0) {
      MS_LOG(DEBUG) << "Precedent task " << pre_task->name() << " does not have output tensors";
      continue;
    }
    if (prenode_index.second > pre_task->out_tensors().size()) {
      MS_LOG(DEBUG) << "Output index " << prenode_index.second << " exceeds input node ["
                    << prenode_index.first->fullname_with_scope() << "]'s outputs size "
                    << pre_task->out_tensors().size();
      continue;
    }
    auto input_tensor = pre_task->out_tensors()[prenode_index.second];
    MS_EXCEPTION_IF_NULL(input_tensor);
    input_tensor->consumers().insert(task);
    task->in_tensors().insert(input_tensor);
    MS_LOG(DEBUG) << "GptoTensor " << input_tensor->id() << " has new consumer " << task->id();
    if (input_tensor->type() == GptoTensorType::kWorkspace) {
      input_tensor->set_type(GptoTensorType::kSimple);
    }
  }
}

void ExtractRealTensors(const SchedulingInput &scheduling_input,
                        std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);
  const auto &tasks = scheduling_input.tasks;

  ExtractOutputWorkspaceTensors(scheduling_input, tasks);

  // Looping over tasks to obtain input tensors after all outputs have been saved
  static std::unordered_set<void *> parameter_set;
  for (auto &task : tasks) {
    const auto &kernel = task->cnode();
    const auto &kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    if (common::AnfAlgo::GetCNodeName(kernel) != kMemSetOpName) {  // standard input case
      StandardInputCase(task, &parameter_set, kernel, cnode_to_task_gpto_map_ptr);
    } else {  // atomic clean input case
      auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_tensor_num; i++) {
        auto pre_node = kernel->input(i + 1)->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(pre_node);

        auto iter = cnode_to_task_gpto_map_ptr->find(pre_node);
        if (iter == cnode_to_task_gpto_map_ptr->end()) {
          MS_LOG(DEBUG) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                        << pre_node->fullname_with_scope() << "] not found in tasks";
          continue;
        }
        auto pre_task = iter->second;

        if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {  // clean output
          auto clean_output_indexs =
            common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
          for (auto index : clean_output_indexs) {
            CleanOutput(index, pre_node, pre_task, task);
          }
        }

        if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {  // clean workspace
          CleanWorkspace(pre_node, pre_task, task);
        }
      }
    }
  }
  parameter_set.clear();
}

size_t CalculateVectorCost(const CNodePtr &cnode) {
  Time cost = 0;
  if (common::AnfAlgo::GetInputTensorNum(cnode) == 0) {
    return cost;
  }
  KernelWithIndex kernel_with_index_1 = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
  ShapeVector shape_1 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_1.first, kernel_with_index_1.second);
  const TypeId type_1 = common::AnfAlgo::GetOutputInferDataType(kernel_with_index_1.first, 0);
  size_t type_size_1 = GetDataTypeSize(type_1);
  size_t compute_count = std::accumulate(shape_1.cbegin(), shape_1.cend(), 1, std::multiplies<size_t>{});
  const double kLatency = 0.5;
  const size_t kParallel = 128;
  cost = kLatency + (compute_count * type_size_1 / kParallel);
  return cost;
}

size_t CalculateCubeCost(const CNodePtr &cnode) {
  Time cost = 0;
  // Get info of input 1
  size_t batch = 1;
  KernelWithIndex kernel_with_index_1 = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
  ShapeVector shape_1 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_1.first, kernel_with_index_1.second);

  // Get info of input 2
  KernelWithIndex kernel_with_index_2 = common::AnfAlgo::GetPrevNodeOutput(cnode, 1);
  ShapeVector shape_2 = common::AnfAlgo::GetOutputInferShape(kernel_with_index_2.first, kernel_with_index_2.second);

  // Get info of output
  ShapeVector shape_out = common::AnfAlgo::GetOutputInferShape(cnode, 0);

  // Remove batch if operator is batchmatmul
  const size_t kShapeSizeFour = 4;
  if (IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul) || IsPrimitiveCNode(cnode, prim::kPrimBatchMatMulV2)) {
    batch = shape_1.front();
    if (shape_1.size() == kShapeSizeFour) {
      shape_1.erase(shape_1.begin());
      shape_1.erase(shape_1.begin());
      shape_out.erase(shape_out.begin());
      shape_out.erase(shape_out.begin());
    } else {
      shape_1.erase(shape_1.begin());
      shape_2.erase(shape_2.begin());
      shape_out.erase(shape_out.begin());
    }
  }

  // Find MKN
  size_t k = 0;
  size_t m = 0;
  size_t n = 0;
  std::vector<size_t> tmp;
  std::copy(shape_1.begin(), shape_1.end(), back_inserter(tmp));
  for (auto dim : shape_2) {
    bool found_in_input = std::find(tmp.begin(), tmp.end(), dim) != tmp.end();
    bool found_in_output = std::find(shape_out.begin(), shape_out.end(), dim) != shape_out.end();
    if (found_in_input && k == 0 && !found_in_output) {
      k = dim;
      tmp.erase(std::remove(tmp.begin(), tmp.end(), dim), tmp.end());
    } else if (found_in_input && k == 0 && found_in_output && n != 0) {
      k = dim;
    } else {
      n = dim;
    }
  }
  m = tmp[0];

  // Get info of dtype
  const TypeId type_1 = common::AnfAlgo::GetOutputInferDataType(kernel_with_index_1.first, 0);
  size_t type_size_1 = GetDataTypeSize(type_1);
  const Time kLatency = 21;
  const Time kParallel = 8192;
  cost = kLatency + batch * m * k * n * type_size_1 / kParallel;
  return cost;
}

size_t CalculateCommCost(const CNodePtr &cnode) {
  Time cost = 0;
  size_t output_num = AnfUtils::GetOutputTensorNum(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);

  // For each operator we get the inputs and outputs
  // For each inputs, we multiply the shape to have the total size and we multiply the size by the data type
  // We then sum all inputs
  // If there is more than 1 output, we do the same for the outputs
  // If output == 1 then cost is 0. We then sum all outputs
  // We sum inputs cost + outputs cost
  for (size_t j = 0; j < input_num; j++) {
    KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, j);
    if (dyn_cast<abstract::BaseShape>(kernel_with_index.first->Shape()) == nullptr ||
        dyn_cast<Type>(kernel_with_index.first->Type()) == nullptr) {
      MS_LOG(DEBUG) << "shape or type is nullptr, ignore";
      continue;
    }
    ShapeVector shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
    if (shape.size() <= 0) {
      continue;
    }

    const TypeId type = common::AnfAlgo::GetOutputInferDataType(kernel_with_index.first, 0);
    if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) {
      continue;
    }

    size_t type_size = GetDataTypeSize(type);
    cost += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
  }

  if (output_num > 1) {
    for (size_t j = 0; j < output_num; j++) {
      ShapeVector shape = common::AnfAlgo::GetOutputInferShape(cnode, j);
      if (shape.size() <= 0) {
        continue;
      }

      const TypeId type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
      if (type == kObjectTypeUMonad || type == kObjectTypeMonad || type == kObjectTypeFunction) {
        continue;
      }

      size_t type_size = GetDataTypeSize(type);
      cost += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
    }
  }

  return cost;
}

std::pair<Time, Memory> LogBaseline(std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr,
                                    const KernelGraphPtr &kernel_graph, const std::string &path, bool print_log_file) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_gpto_map_ptr);

  const size_t graph_id = kernel_graph->graph_id();
  std::ostringstream oss;
  std::string filename;

  if (print_log_file) {
    std::stringstream ss;
    ss << kernel_graph;
    filename = path + std::string("/") + std::string("gpto_baseline_") + std::to_string(graph_id) + std::string("_") +
               ss.str() + std::string(".log");
  }

  std::unordered_map<GptoTaskId, Time> taskid_to_end_value;
  std::unordered_map<GptoTaskId, Time> taskid_to_start_value;
  size_t makespan = 0;
  const std::vector<CNodePtr> execution_order = kernel_graph->execution_order();
  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];

    GptoTaskPtr current_task = (*cnode_to_task_gpto_map_ptr)[cnode];

    // Find the latest executed task which has the same type as the current task
    GptoTaskPtr last_task = nullptr;
    MS_LOG(DEBUG) << "Current value loop: " << i << " with node: " << execution_order[i]->UniqueName();
    for (int j = i - 1; j >= 0; j--) {
      MS_LOG(DEBUG) << "Current value loop j: " << j;
      GptoTaskPtr tmp_task = (*cnode_to_task_gpto_map_ptr)[execution_order[j]];
      MS_LOG(DEBUG) << "With node: " << tmp_task->name();
      if (tmp_task->gpto_type() == current_task->gpto_type()) {
        MS_LOG(DEBUG) << "Found node same type";
        last_task = tmp_task;
        break;
      }
    }

    // Find the latest parent of the current task
    for (const auto &parent : (*cnode_to_task_gpto_map_ptr)[cnode]->parents()) {
      if (last_task == nullptr || taskid_to_end_value[parent.lock()->id()] >= taskid_to_end_value[last_task->id()]) {
        last_task = parent.lock();
        MS_LOG(DEBUG) << "Found parent " << last_task->name();
      }
    }

    if (last_task == nullptr) {
      last_task = current_task;
      taskid_to_start_value[current_task->id()] = 0;
      taskid_to_end_value[current_task->id()] = 0 + current_task->cost();
    } else {
      taskid_to_start_value[current_task->id()] = taskid_to_end_value[last_task->id()];
      taskid_to_end_value[current_task->id()] = taskid_to_start_value[current_task->id()] + current_task->cost();
    }

    size_t current_task_end = taskid_to_end_value[current_task->id()];
    if (current_task_end > makespan) {
      makespan = taskid_to_end_value[current_task->id()];
    }

    if (print_log_file) {
      oss << "TASK id=" << std::to_string(current_task->id()) << ", name=" << current_task->name()
          << ", type=" << std::to_string(current_task->gpto_type()) << ", cost=" << std::to_string(current_task->cost())
          << ", start=" << std::to_string(taskid_to_start_value[current_task->id()])
          << ", end=" << std::to_string(current_task_end) << "\n";
    }
  }
  MS_LOG(INFO) << "Makespan estimate of baseline is " + std::to_string(makespan);

  if (print_log_file) {
    (void)Common::SaveStringToFile(filename, oss.str());
  }

  Memory memory =
    MemoryEstimateBaseline(execution_order, cnode_to_task_gpto_map_ptr, &taskid_to_start_value, &taskid_to_end_value);
  return std::make_pair(makespan, memory);
}

Memory MemoryEstimateBaseline(const std::vector<CNodePtr> &execution_order,
                              std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_gpto_map_ptr,
                              std::unordered_map<GptoTaskId, Time> *taskid_to_start_value_ptr,
                              std::unordered_map<GptoTaskId, Time> *taskid_to_end_value_ptr) {
  std::map<Time, Memory> mem_peak;
  Memory graph_output_mem = 0;
  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];
    GptoTaskPtr task = (*cnode_to_task_gpto_map_ptr)[cnode];
    mem_peak[(*taskid_to_start_value_ptr)[task->id()]] = 0;
    mem_peak[(*taskid_to_end_value_ptr)[task->id()]] = 0;
  }

  for (size_t i = 0; i < execution_order.size(); i++) {
    const auto &cnode = execution_order[i];
    const GptoTaskPtr task = (*cnode_to_task_gpto_map_ptr)[cnode];
    std::vector<GptoTensorPtr> tensors;
    tensors.reserve(task->out_tensors().size() + task->workspace_tensors().size());
    std::copy(task->out_tensors().begin(), task->out_tensors().end(), std::back_inserter(tensors));
    std::copy(task->workspace_tensors().begin(), task->workspace_tensors().end(), std::back_inserter(tensors));
    for (auto &tensor : tensors) {
      const auto &start = (*taskid_to_start_value_ptr)[task->id()];
      Time end = 0;
      if (tensor->consumers().size() > 0) {
        for (auto &consumer : tensor->consumers()) {
          end = std::max(end, (*taskid_to_end_value_ptr)[consumer.lock()->id()]);
        }
      } else {
        end = (*taskid_to_end_value_ptr)[task->id()];
      }
      if (tensor->type() == kGraphOutput) {
        graph_output_mem += tensor->weight();
      }
      for (auto &time_mem : mem_peak) {
        if ((start <= time_mem.first && time_mem.first < end && tensor->type() != kGraphOutput) ||
            (start <= time_mem.first && tensor->type() == kGraphOutput)) {
          mem_peak[time_mem.first] += tensor->weight();
        }
      }
    }
  }
  mem_peak.rbegin()->second -= graph_output_mem;
  if (mem_peak.rbegin()->second > 0) {
    MS_LOG(WARNING) << "Memory peak " << mem_peak.rbegin()->second << " at makespan time " << mem_peak.rbegin()->first;
  }

  Memory memory = std::max_element(mem_peak.begin(), mem_peak.end(),
                                   [](const std::pair<Time, Memory> &a, const std::pair<Time, Memory> &b) -> bool {
                                     return a.second < b.second;
                                   })
                    ->second;
  MS_LOG(INFO) << "Memory estimate of baseline is " << memory;
  return memory;
}

void InitializeTaskInlineCondition(const CNodePtr &cnode, GptoTaskPtr *task,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *switch_attribute_ids,
                                   std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> *gather_attribute_ids) {
  MS_EXCEPTION_IF_NULL(task);
  MS_EXCEPTION_IF_NULL(switch_attribute_ids);
  MS_EXCEPTION_IF_NULL(gather_attribute_ids);

  const size_t kPrefixLength = 12;
  if (cnode->HasAttr(kInlineSubGraphName)) {  // ConditionSwitch
    (*task)->set_condition_switch(true);
    std::string s = cnode->GetAttr(kInlineSubGraphName)->ToString();
    std::string s1 = s.substr(s.find('(') + 1, s.find(',') - 1);
    std::string s2 = s.substr(s.find(',') + 1, s.find(')') - 1);
    (*switch_attribute_ids)[(*task)] = std::make_pair(std::stoll(s1.substr(s1.find("kernel_graph") + kPrefixLength)),
                                                      std::stoll(s2.substr(s2.find("kernel_graph") + kPrefixLength)));
    MS_LOG(DEBUG) << "Task ConditionSwitch " << (*task)->id() << " with attribute kInlineSubGraphName" << s;
  } else if (cnode->HasAttr(kAttrBranchGraphName)) {  // ConditionGather
    (*task)->set_condition_gather(true);
    std::string s = cnode->GetAttr(kAttrBranchGraphName)->ToString();
    std::string s1 = s.substr(s.find('(') + 1, s.find(',') - 1);
    std::string s2 = s.substr(s.find(',') + 1, s.find(')') - 1);
    (*gather_attribute_ids)[(*task)] = std::make_pair(std::stoll(s2.substr(s2.find("kernel_graph") + kPrefixLength)),
                                                      std::stoll(s1.substr(s1.find("kernel_graph") + kPrefixLength)));
    MS_LOG(DEBUG) << "Task ConditionGather " << (*task)->id() << " with attribute kAttrBranchGraphName" << s;
  }
}

void PushTasksToVisit(std::queue<std::weak_ptr<GptoTask>> *tasks_to_visit,
                      std::unordered_map<size_t, size_t> *unprocessed_children, const std::weak_ptr<GptoTask> &parent,
                      GptoTaskPtr switch_task, size_t count_condition) {
  (*unprocessed_children)[parent.lock()->id()] -= 1;
  if ((*unprocessed_children)[parent.lock()->id()] == 0) {
    if (parent.lock() != switch_task) {
      tasks_to_visit->push(parent);
    } else {
      if (switch_task->subgraph_id() == count_condition) {
        switch_task->set_subgraph_id(count_condition + 1);
        MS_LOG(DEBUG) << "Assign subgraph id " << count_condition << " to task " << switch_task->id() << " name "
                      << switch_task->name();
      }
    }
  }
}

void UpdateTasksInlineCondition(std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr,
                                std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> *switch_gather) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  MS_EXCEPTION_IF_NULL(switch_gather);

  size_t count_condition = SIZE_MAX - 1;
  std::unordered_map<size_t, size_t> unprocessed_children;
  std::queue<std::weak_ptr<GptoTask>> tasks_to_visit;

  for (const auto &key_val : *cnode_to_task_map_ptr) {
    auto &task = key_val.second;
    unprocessed_children[task->id()] = task->children().size();
  }

  for (auto &it : (*switch_gather)) {
    const auto &switch_task = it.first;
    const auto &gather_task = it.second;
    MS_LOG(INFO) << "Assign subgraph id " << count_condition << " to tasks under ConditionSwitch task "
                 << switch_task->id() << " name " << switch_task->name()
                 << " up to (and including) ConditionGather task " << gather_task->id() << " name "
                 << gather_task->name();
    gather_task->set_subgraph_id(count_condition);

    for (auto parent : gather_task->parents()) {
      if (parent.lock() == switch_task) {
        parent.lock()->set_subgraph_id(count_condition + 1);
        MS_LOG(INFO) << "Assign subgraph id " << count_condition + 1 << " to task " << switch_task->id() << " name "
                     << switch_task->name();
      } else {
        tasks_to_visit.push(parent);
      }
    }

    while (!tasks_to_visit.empty()) {
      const auto &selected_task = tasks_to_visit.front().lock();
      selected_task->set_subgraph_id(count_condition);

      if (selected_task->name().find("ConditionGather") != std::string::npos) {
        // Get the switch node of the nested gather node
        GptoTaskPtr selected_task_switch =
          std::find_if(switch_gather->begin(), switch_gather->end(), [selected_task](const auto &i) {
            return i.second->name() == selected_task->name();
          })->first;
        // Get all nested switch parents to visit
        for (const auto switch_parent : selected_task_switch->parents()) {
          PushTasksToVisit(&tasks_to_visit, &unprocessed_children, switch_parent, switch_task, count_condition);
        }
      } else {
        for (const auto &parent : selected_task->parents()) {
          PushTasksToVisit(&tasks_to_visit, &unprocessed_children, parent, switch_task, count_condition);
        }
      }
      tasks_to_visit.pop();
    }
    gather_task->set_subgraph_id_parent(switch_task->subgraph_id());

    count_condition--;
  }
}

std::unordered_map<std::string, Time> GetProfCost() {
  std::unordered_map<std::string, Time> profiling_map;
  std::ifstream file;
  file.open(common::GetEnv("MS_ENABLE_GPTO_PROFILING_AS_COST"));
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream s(line);
    std::string field;
    std::vector<std::string> fields;
    while (getline(s, field, ',')) {
      fields.push_back(field);
    }
    if (fields[kIndex0] == "name") {
      continue;
    }
    profiling_map[fields[kIndex0]] = stoi(fields[kIndex1]);
  }
  return profiling_map;
}

SchedulingInput ExtractSchedulingInput(mindspore::device::DeviceResManager *device_res_manager,
                                       const KernelGraphPtr &kernel_graph,
                                       std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr,
                                       [[maybe_unused]] std::map<std::string, size_t> *group_to_id) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  SchedulingInput scheduling_input;  // to fill in and return
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> switch_attribute_ids;
  std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>> gather_attribute_ids;
  GptoTaskPtr getnext_task;

  // Create a task per node
  MS_LOG(INFO) << "Start Extract GPTO Tasks";
  const auto &real_kernels = kernel_graph->execution_order();
  scheduling_input.tasks.reserve(real_kernels.size());

  std::unordered_map<std::string, Time> profiling_map;
  if (common::GetEnv("MS_ENABLE_GPTO_PROFILING_AS_COST") != "") {
    profiling_map = GetProfCost();
  }

  for (size_t i = 0; i < real_kernels.size(); ++i) {
    const auto &cnode = real_kernels[i];

    if (common::AnfAlgo::IsDynamicShape(cnode) || common::AnfAlgo::IsDynamicSequence(cnode) ||
        common::AnfAlgo::IsDynamicValue(cnode)) {
      MS_LOG(INFO) << "GPTO can't parse graph with dynamic shape or dynamic value now.";
      scheduling_input.tasks.clear();
      return scheduling_input;
    }

    GptoTaskPtr task = std::make_shared<GptoTask>(
      i, GetRealType(cnode), GetType(device_res_manager, cnode, group_to_id), cnode->fullname_with_scope());
    task->set_cnode(cnode);
    Time cost = 0;

    if (cnode->UniqueName().find("GetNext") != std::string::npos) {
      getnext_task = task;
    }

    if (common::GetEnv("MS_ENABLE_GPTO_NO_COST") == "1") {
      cost = 1;
    } else if (common::GetEnv("MS_ENABLE_GPTO_PROFILING_AS_COST") != "") {
      std::string node_name(cnode->UniqueName().substr(0, cnode->UniqueName().rfind("_")));
      if (profiling_map.find(node_name) != profiling_map.end()) {
        cost = profiling_map[node_name];
      } else {
        MS_LOG(WARNING) << "Node " << node_name << " not found in profiling file";
      }
    } else if (task->real_type() == kComp) {  // comp node of type Vector
      cost = CalculateVectorCost(cnode);
    } else if (task->real_type() == kCube) {  // comp node of type Cube
      cost = CalculateCubeCost(cnode);
    } else {  // comm node
      cost = CalculateCommCost(cnode);
    }

    task->AssignCost(cost);

    // Start Step 1 ConditionSwitch/Gather for inline: save attributes
    InitializeTaskInlineCondition(cnode, &task, &switch_attribute_ids, &gather_attribute_ids);
    // End Step 1 ConditionSwitch/Gather for inline

    MS_LOG(DEBUG) << "Task " << task->id() << " with name " << cnode->UniqueName() << " and CNodePtr " << cnode
                  << " with cost " << task->cost() << " and type " << task->gpto_type();
    scheduling_input.tasks.push_back(task);
    (*cnode_to_task_map_ptr)[cnode] = task;
  }

  if (gpto_mode == kCompCommGroup) {
    for (auto &group : *group_to_id) {
      MS_LOG(INFO) << "Get group id " << group.second << " with name " << group.first;
    }
    MS_LOG(INFO) << "End Extract GPTO Tasks with " << group_to_id->size() << " comm groups";
  } else {
    MS_LOG(INFO) << "End Extract GPTO Tasks";
  }

  MS_LOG(INFO) << "Start Extract GPTO Edges";
  InsertEdges(kernel_graph, cnode_to_task_map_ptr);
  MS_LOG(INFO) << "End Extract GPTO Edges";

  MS_LOG(INFO) << "Start Extract GPTO Tensors";
  ExtractRealTensors(scheduling_input, cnode_to_task_map_ptr);
  MS_LOG(INFO) << "End Extract GPTO Tensors";

  // Start Step 2 ConditionSwitch/Gather for inline: identify matching switch/gather pairs
  MS_LOG(INFO) << "Start Extract GPTO Switch/Gather";
  ComputeDepthAndTopLevel(scheduling_input.tasks);  // if kept here, do not call again later in Process()

  std::map<GptoTaskPtr, GptoTaskPtr, TaskDepthSort> switch_gather;
  for (auto &switch_it : switch_attribute_ids) {
    const auto &switch_task = switch_it.first;
    auto switch_pair = switch_it.second;

    std::unordered_map<GptoTaskPtr, std::pair<size_t, size_t>>::iterator gather_it;
    for (gather_it = gather_attribute_ids.begin(); gather_it != gather_attribute_ids.end(); ++gather_it) {
      if (gather_it->second == switch_pair) {
        break;
      }
    }
    if (gather_it == gather_attribute_ids.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Could not find matching ConditionGather for a given ConditionSwitch "
                                 << switch_pair;
    }
    const auto &gather_task = gather_it->first;
    switch_gather[switch_task] = gather_task;
    MS_LOG(DEBUG) << "Mapped ConditionSwitch task " << switch_task->id() << " to ConditionGather task "
                  << gather_task->id();
  }
  MS_LOG(INFO) << "End Extract GPTO Switch/Gather";
  // End Step 2 ConditionSwitch/Gather for inline

  // Start Step 3 ConditionSwitch/Gather for inline: traverse each Condition/Switch gather block to assign proper ids
  // Assumption 1: switch and nodes before gather have no predecessors/descendants outside the block
  // Assumption 2: conditional switch does not have conditional gather as a child
  MS_LOG(INFO) << "Start Update Inline";
  UpdateTasksInlineCondition(cnode_to_task_map_ptr, &switch_gather);
  MS_LOG(INFO) << "End Update Inline";
  // End Step 3 ConditionSwitch/Gather for inline

  MakeRootGetNext(getnext_task, cnode_to_task_map_ptr);

  return scheduling_input;
}

void MakeRootGetNext(const GptoTaskPtr &getnext_task,
                     std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  if (getnext_task == nullptr) {
    return;
  }

  for (const auto &key_val : *cnode_to_task_map_ptr) {
    auto &task = key_val.second;
    if (task->parents().size() != 0 || task == getnext_task) {
      continue;
    }
    task->AddParent(getnext_task);
    getnext_task->AddChild(task);
  }
}

// Calculate the lower bound for a single task
Memory CalculateTaskLowerBound(const GptoTaskPtr &task, const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors,
                               const std::vector<mindspore::somas::DynamicBitSet> &nodes_dependency) {
  Memory task_lb = 0;
  for (const auto &tensor : tensors) {
    if (tensor->weight() == 0) {
      continue;
    }
    const auto &source = tensor->source().lock();
    const auto &consumers = tensor->consumers();

    if (task == source || consumers.count(task) > 0) {
      task_lb += tensor->weight();
    } else {
      if (nodes_dependency[task->id()].IsBitTrue(source->id())) {
        if (tensor->type() == kGraphOutput) {
          task_lb += tensor->weight();
        } else {
          if (std::any_of(consumers.cbegin(), consumers.cend(), [&](auto &consumer) {
                return nodes_dependency[consumer.lock()->id()].IsBitTrue(task->id());
              })) {
            task_lb += tensor->weight();
          }
        }
      }
    }
  }
  return task_lb;
}

// Calculate the maximum lower bound across all tasks
Memory MemoryLowerBound(const std::vector<GptoTaskPtr> &tasks,
                        const std::vector<mindspore::somas::DynamicBitSet> &nodes_dependency,
                        const std::set<GptoTensorPtr, GptoTensorIdSort> &tensors) {
  Memory max_lb = 0;
  for (const auto &task : tasks) {
    Memory task_lb = CalculateTaskLowerBound(task, tensors, nodes_dependency);
    task->set_lower_bound(task_lb);
    max_lb = std::max(max_lb, task_lb);
  }
  MS_LOG(INFO) << "Memory Lower bound for tensors " << max_lb << " (" << max_lb / kMBToByte << " MB)";
  return max_lb;
}

void GraphOutputProcess(const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);

  size_t count = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  auto &cnode_to_task_map = *cnode_to_task_map_ptr;
  for (auto &output : outputs) {
    auto output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto output_kernel = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_kernel);
    while (AnfUtils::IsRealCNodeKernel(output_kernel) &&
           cnode_to_task_map.find(output_kernel->cast<CNodePtr>()) == cnode_to_task_map.end()) {
      auto cnode = output_kernel->cast<CNodePtr>();
      if (!common::AnfAlgo::IsNopNode(cnode)) {
        MS_LOG(INTERNAL_EXCEPTION) << "Node[" << cnode->fullname_with_scope()
                                   << "] doesn't exist in cnode_to_task_map and is not a nop node!!!";
      }
      output_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
      output_kernel = output_with_index.first;
    }

    if (!AnfUtils::IsRealCNodeKernel(output_kernel)) {
      continue;
    }

    auto output_index = output_with_index.second;
    auto iter = cnode_to_task_map.find(output_kernel->cast<CNodePtr>());
    if (iter != cnode_to_task_map.end()) {
      auto &node = iter->second;
      MS_EXCEPTION_IF_NULL(node);
      if (node->out_tensors().size() == 0) {
        MS_LOG(DEBUG) << "Node " << node->name() << " does not have output tensors";
        continue;
      } else if (output_index < node->out_tensors().size()) {
        auto &tensor = node->out_tensors()[output_index];
        tensor->set_type(kGraphOutput);  // if need_reuse_graph_output (default is true), then treat as semilifelong
                                         // end, otherwise set to 0
        MS_LOG(DEBUG) << "GPTO Tensor " << tensor->id() << " with size " << tensor->weight() << " is kGraphOutput";
        count++;
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Graph's output node " << output_kernel->fullname_with_scope()
                                   << "'s output index " << output_index << " is larger than its output tensor number "
                                   << node->out_tensors().size();
      }
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Can't find task for graph output node " << output_kernel->fullname_with_scope();
    }
  }
  MS_LOG(INFO) << "Found " << count << " graph output tensors for GPTO";
}

namespace {  // somas-like functionality for merging ref node (assuming unions of pairs of tensors originally)
size_t find_father(std::vector<size_t> *father, size_t x) {
  MS_EXCEPTION_IF_NULL(father);
  if (x >= father->size()) {
    MS_LOG(EXCEPTION) << "Index " << x << " out of range " << father->size();
  }
  if (x == (*father)[x]) {
    return x;
  }
  (*father)[x] = find_father(father, (*father)[x]);
  return (*father)[x];
}

std::vector<std::vector<GptoTensorPtr>> GetRegularUnionTensorsList(
  size_t max_tensor, const std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> &union_tensors) {
  std::vector<size_t> father;
  std::map<size_t, GptoTensorPtr> id_to_tensor;
  for (size_t i = 0; i < max_tensor; i++) {
    father.push_back(i);
  }
  for (auto union_pair : union_tensors) {
    const size_t &tid0 = union_pair.first->id();
    const size_t &tid1 = union_pair.second->id();
    father[find_father(&father, tid1)] = find_father(&father, tid0);
    id_to_tensor[tid0] = union_pair.first;
    id_to_tensor[tid1] = union_pair.second;
  }

  std::map<size_t, size_t> kv;
  std::vector<std::vector<GptoTensorPtr>> ret_union_list;
  std::vector<std::set<size_t>> union_tensor_sets;
  for (const auto &union_pair : union_tensors) {
    std::vector<size_t> tids = {union_pair.first->id(), union_pair.second->id()};
    for (const auto &tid : tids) {
      size_t fa = find_father(&father, tid);
      if (kv.find(fa) == kv.end()) {
        ret_union_list.emplace_back();
        union_tensor_sets.emplace_back();
        kv.emplace(fa, ret_union_list.size() - 1);
      }
      auto &union_tensor_set = union_tensor_sets[kv.at(fa)];
      if (union_tensor_set.find(tid) == union_tensor_set.end()) {
        ret_union_list[kv.at(fa)].push_back(id_to_tensor[tid]);
        union_tensor_set.insert(tid);
      }
    }
  }
  return ret_union_list;
}
}  // namespace

std::map<GptoTensorPtr, GptoTensorPtr> GetRefTensorsInContiguousList(
  const std::vector<std::vector<GptoTensorPtr>> &ref_node_vector) {
  std::map<GptoTensorPtr, GptoTensorPtr> ref_tensors_in_contiguous_map;  // key: refnode input, value: refnode output
  constexpr auto kRefNodeTensorNum = 2;
  for (auto ref_list : ref_node_vector) {
    auto contiguous_in_ref_list = std::count_if(ref_list.begin(), ref_list.end(), [](GptoTensorPtr t) {
      MS_EXCEPTION_IF_NULL(t);
      return t->contiguous();
    });
    if (ref_list.size() > kRefNodeTensorNum && contiguous_in_ref_list > 0) {
      MS_LOG(WARNING) << "Ref node of size greater than two with at least one contiguous tensor in";
    }
    if (ref_list.size() == kRefNodeTensorNum && contiguous_in_ref_list == 1) {
      MS_EXCEPTION_IF_NULL(ref_list[0]);
      MS_EXCEPTION_IF_NULL(ref_list[1]);
      MS_LOG(WARNING) << "Ref node of size two with only one contiguous tensor" << ref_list[0]->id() << ":"
                      << ref_list[0]->contiguous() << ", " << ref_list[1]->id() << ":" << ref_list[1]->contiguous();
    }
    if (ref_list.size() == kRefNodeTensorNum && static_cast<size_t>(contiguous_in_ref_list) == kRefNodeTensorNum) {
      ref_tensors_in_contiguous_map[ref_list[0]] = ref_list[1];
    }
  }
  return ref_tensors_in_contiguous_map;
}

std::map<size_t, size_t> GetContiguousRefIndexMap(
  const std::vector<std::vector<GptoTensorPtr>> &ref_node_vector,
  const std::vector<std::vector<GptoTensorPtr>> &contiguous_tensors_list) {
  std::map<size_t, size_t> contiguous_list_with_ref_index_map;
  std::map<GptoTensorPtr, GptoTensorPtr> ref_tensors_in_contiguous_map = GetRefTensorsInContiguousList(ref_node_vector);
  for (const auto &ref_pair : ref_tensors_in_contiguous_map) {
    const auto &ref_first = ref_pair.first;
    const auto &ref_second = ref_pair.second;
    bool found_first = false;
    bool found_second = false;
    size_t index_first = 0;
    size_t index_second = 0;
    size_t index_in_list_first = 0;
    size_t index_in_list_second = 0;
    for (size_t index = 0; index < contiguous_tensors_list.size() && (!found_first || !found_second); index++) {
      if (!found_first) {
        auto iterator_first =
          std::find(contiguous_tensors_list[index].begin(), contiguous_tensors_list[index].end(), ref_first);
        if (iterator_first != contiguous_tensors_list[index].end()) {
          index_first = index;
          index_in_list_first = static_cast<size_t>(iterator_first - contiguous_tensors_list[index].begin());
          found_first = true;
        }
      }
      if (!found_second) {
        auto iterator_second =
          std::find(contiguous_tensors_list[index].begin(), contiguous_tensors_list[index].end(), ref_second);
        if (iterator_second != contiguous_tensors_list[index].end()) {
          index_second = index;
          index_in_list_second = static_cast<size_t>(iterator_second - contiguous_tensors_list[index].begin());
          found_second = true;
        }
      }
    }
    if (!found_first && !found_second) {
      continue;
    }
    if (!found_first) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_first->id() << " not found in any contiguous list";
    }
    if (!found_second) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_second->id() << " not found in any contiguous list";
    }
    if (contiguous_list_with_ref_index_map.find(index_first) == contiguous_list_with_ref_index_map.end() ||
        contiguous_list_with_ref_index_map[index_first] == index_second) {
      contiguous_list_with_ref_index_map[index_first] = index_second;
      // Checking for error cases
      if (index_in_list_first != index_in_list_second) {
        MS_LOG(WARNING) << "Inconsistency in contiguous ref: tensor " << ref_first->id() << " in position "
                        << index_in_list_first << " of contiguous list " << index_first << " and tensor "
                        << ref_second->id() << " in position " << index_in_list_second << " of contiguous list "
                        << index_second;
      }
    } else {
      MS_LOG(WARNING) << "Contiguous list " << index_first << " associated (ref node) with two other contiguous lists: "
                      << contiguous_list_with_ref_index_map[index_first] << " and " << index_second;
    }
  }
  return contiguous_list_with_ref_index_map;
}

std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> RefNodeProcess(
  const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  auto &cnode_to_task_map = *cnode_to_task_map_ptr;
  const auto &kernel_cnodes = graph->execution_order();
  size_t total_output_size = 0;
  size_t total_input_size = 0;
  std::vector<std::pair<GptoTensorPtr, GptoTensorPtr>> in_out_vector;

  // Loop to obtain ref node pairs
  for (const auto &kernel : kernel_cnodes) {
    auto mod = AnfAlgo::GetKernelMod(kernel);
    if (mod == nullptr) {
      MS_LOG(WARNING) << "Null mod for kernel " << kernel->fullname_with_scope();
      continue;
    }
    size_t index = 0;
    for (const auto &size : mod->GetOutputSizeList()) {
      auto out_index = index++;
      session::AnfWithOutIndex out_pair(kernel, out_index);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto &node = cnode_to_task_map[kernel];
        MS_EXCEPTION_IF_NULL(node);
        auto output_tensor = node->out_tensors()[out_index];
        MS_EXCEPTION_IF_NULL(output_tensor);
        total_output_size += size;

        if (!AnfUtils::IsRealCNodeKernel(origin_pair.first)) {
          output_tensor->set_type(kGraphInput);
          output_tensor->set_weight(0);
          continue;
        }

        if (cnode_to_task_map.find(origin_pair.first->cast<CNodePtr>()) == cnode_to_task_map.end()) {
          auto cnode = origin_pair.first->cast<CNodePtr>();
          if (!common::AnfAlgo::IsNopNode(cnode)) {
            MS_LOG(INTERNAL_EXCEPTION) << "Node[" << origin_pair.first->fullname_with_scope() << "] find input node["
                                       << cnode->fullname_with_scope()
                                       << "] doesn't exist in nodes_map and is not a nop node!!!!";
          }
          origin_pair = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(1), 0, false);
        }
        if (!origin_pair.first->isa<CNode>()) {
          MS_LOG(INTERNAL_EXCEPTION) << "The origin_pair.first is not a cnode. Info origin_pair.first: "
                                     << origin_pair.first->DebugString();
        }
        auto ori_node = origin_pair.first->cast<CNodePtr>();
        auto ori_index = origin_pair.second;
        if (cnode_to_task_map.find(ori_node.get()->cast<CNodePtr>()) == cnode_to_task_map.end()) {
          MS_LOG_WITH_NODE(EXCEPTION, ori_node)
            << "The ori_node is not included in cnode_to_task_map constructed from exec_order of graph. Info ori_node: "
            << ori_node->DebugString();
        }
        auto &repeat_node = cnode_to_task_map[ori_node];
        MS_EXCEPTION_IF_NULL(repeat_node);
        auto input_tensor = repeat_node->out_tensors()[ori_index];
        MS_EXCEPTION_IF_NULL(input_tensor);
        in_out_vector.push_back(std::make_pair(input_tensor, output_tensor));
        total_input_size += input_tensor->weight();
        MS_LOG(DEBUG) << "RefNode: input " << input_tensor->id() << " output " << output_tensor->id();
      }
    }
  }
  MS_LOG(INFO) << "RefNode tensor total size: input " << total_input_size << " output " << total_output_size;
  return in_out_vector;
}

std::vector<std::vector<GptoTensorPtr>> CommunicationNodeProcess(
  std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_to_task_map_ptr);
  const auto &cnode_to_task_map = *cnode_to_task_map_ptr;

  std::vector<std::vector<GptoTensorPtr>> contiguous_tensors_list;
  Memory comm_input_total_size = 0;
  Memory comm_output_total_size = 0;
  for (const auto &cnode_task : cnode_to_task_map) {
    const auto &task = cnode_task.second;
    MS_EXCEPTION_IF_NULL(task);
    if (task->gpto_type() < kComm || task->name().find(kMatMulAllReduceOpName) != std::string::npos) {
      continue;
    }

    // Contiguous input
    if (task->in_tensors().size() > 1 && (*(task->in_tensors().begin()) != nullptr) &&
        (!(*(task->in_tensors().begin()))->contiguous())) {
      std::vector<GptoTensorPtr> inputs;
      for (const auto &in_tensor : task->in_tensors()) {
        MS_EXCEPTION_IF_NULL(in_tensor);
        if (in_tensor->weight() > 0) {
          MS_EXCEPTION(ValueError) << "The size of communication in_tensor is zero, tensor id: " +
                                        std::to_string(in_tensor->id());
        }
        comm_input_total_size += in_tensor->weight();
        in_tensor->set_contiguous(true);
        inputs.push_back(in_tensor);
      }
      contiguous_tensors_list.push_back(inputs);
    }

    // Contiguous output
    if (task->out_tensors().size() > 1 && (task->out_tensors()[0] != nullptr) &&
        (!task->out_tensors()[0]->contiguous())) {
      std::vector<GptoTensorPtr> outputs;
      for (const auto &out_tensor : task->out_tensors()) {
        MS_EXCEPTION_IF_NULL(out_tensor);
        if (out_tensor->weight() > 0) {
          MS_EXCEPTION(ValueError) << "The size of communication out_tensor is zero, tensor id: " +
                                        std::to_string(out_tensor->id());
        }
        comm_output_total_size += out_tensor->weight();
        out_tensor->set_contiguous(true);
        outputs.push_back(out_tensor);
      }
      if (outputs.size() != (std::set<GptoTensorPtr>(outputs.begin(), outputs.end())).size()) {
        MS_LOG(EXCEPTION) << task->name() << " has duplicate output tensors, please double check task output tensors";
      }
      contiguous_tensors_list.push_back(outputs);
    }

    // Check for duplication of tensors in lists
    std::set<GptoTensorPtr> all_contiguous_tensors_set;
    size_t all_contiguous_tensors_num = 0;
    for (auto &tensors : contiguous_tensors_list) {
      all_contiguous_tensors_num += tensors.size();
      all_contiguous_tensors_set.insert(tensors.begin(), tensors.end());
    }
    if (all_contiguous_tensors_num != all_contiguous_tensors_set.size()) {
      MS_LOG(EXCEPTION) << "Please check communication tasks, some tensors are in multiple contiguous lists";
    }
  }

  MS_LOG(INFO) << "Contiguous tensor total size: input " << comm_input_total_size << " output "
               << comm_output_total_size;
  return contiguous_tensors_list;
}

void UpdateRefNodeGpto(const KernelGraphPtr &graph, std::unordered_map<CNodePtr, GptoTaskPtr> *cnode_to_task_map_ptr) {
  // Retrieve necessary ref node and contiguous info
  auto in_out_vector = RefNodeProcess(graph, cnode_to_task_map_ptr);
  std::vector<std::vector<GptoTensorPtr>> ref_node_vector =
    GetRegularUnionTensorsList(MAX_TENSOR_ID + 1, in_out_vector);
  auto contiguous_tensors_list = CommunicationNodeProcess(cnode_to_task_map_ptr);
  auto contiguous_ref_index_map = GetContiguousRefIndexMap(ref_node_vector, contiguous_tensors_list);

  // Loop to update ref node tensor sizes and gpto graph logic
  for (auto &union_list : ref_node_vector) {
    if (std::any_of(union_list.begin(), union_list.end(), [](const GptoTensorPtr &t) { return t->weight() == 0; })) {
      for (auto &tensor : union_list) {
        tensor->set_weight(0);
      }
    } else {  // update head tensor weight
      const auto &tensor_head = union_list[0];
      for (size_t i = 1; i < union_list.size(); ++i) {
        const auto &tensor_ref = union_list[i];
        if (!tensor_ref->contiguous()) {
          if (tensor_head->weight() < tensor_ref->weight()) {
            tensor_head->set_weight(tensor_ref->weight());
          }
          tensor_ref->set_weight(0);
        }
        for (auto &consumer : tensor_ref->consumers()) {
          tensor_head->consumers().insert(consumer);
          consumer.lock()->in_tensors().insert(tensor_head);
          tensor_head->source().lock()->AddChild(consumer.lock());
          consumer.lock()->AddParent(tensor_head->source().lock());
        }
      }
    }
  }

  // Loop to update ref node for contiguous
  for (auto ref_list_pair : contiguous_ref_index_map) {
    const auto &head_list = contiguous_tensors_list.at(ref_list_pair.first);
    const auto &ref_list = contiguous_tensors_list.at(ref_list_pair.second);
    if (head_list.size() != ref_list.size()) {
      MS_LOG(WARNING) << "Different head list " << ref_list_pair.first << " size " << head_list.size()
                      << " and ref list " << ref_list_pair.second << " size " << ref_list.size();
    }
    for (size_t i = 0; i < head_list.size(); i++) {
      const auto &tensor_head = head_list[i];
      const auto &tensor_ref = ref_list[i];
      MS_EXCEPTION_IF_NULL(tensor_head);
      MS_EXCEPTION_IF_NULL(tensor_ref);
      tensor_ref->set_weight(0);
      for (auto &consumer : tensor_ref->consumers()) {
        tensor_head->consumers().insert(consumer);
        consumer.lock()->in_tensors().insert(tensor_head);
        tensor_head->source().lock()->AddChild(consumer.lock());
        consumer.lock()->AddParent(tensor_head->source().lock());
      }
    }
  }
}

void ExtractTensors(const std::vector<GptoTaskPtr> &tasks, std::set<GptoTensorPtr, GptoTensorIdSort> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  for (const auto &task : tasks) {
    const auto &out_tensors = task->out_tensors();
    const auto &ws_tensors = task->workspace_tensors();
    tensors->insert(out_tensors.begin(), out_tensors.end());
    tensors->insert(ws_tensors.begin(), ws_tensors.end());
  }
}

void MockExecutionOrder(const SchedulingOutput &scheduling_output,
                        std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *mock_exec_order,
                        size_t num_events) {
  std::vector<Interval> task_times = scheduling_output.task_times;
  std::sort(task_times.begin(), task_times.end(), [](Interval x, Interval y) {
    return x.start < y.start || (x.start == y.start && x.task->gpto_type() > y.task->gpto_type());
  });
  mock_exec_order->resize(task_times.size() + 2 * num_events);
  size_t mock_idx = mock_exec_order->capacity() - 1;
  size_t event_id = 0;
  for (std::vector<Interval>::reverse_iterator rit = task_times.rbegin(); rit != task_times.rend(); ++rit) {
    (*mock_exec_order)[mock_idx--] = std::make_pair(rit->task->cnode(), std::make_tuple('n', 0, 0, 0));
    size_t local_event_count = 0;
    for (auto task : rit->task->recv_events()) {
      (*mock_exec_order)[mock_idx - rit->task->recv_events().size() * 2 + 1 + local_event_count] =
        std::make_pair(nullptr, std::make_tuple('s', task.lock()->gpto_type(), rit->task->gpto_type(), event_id));
      (*mock_exec_order)[mock_idx - local_event_count] =
        std::make_pair(nullptr, std::make_tuple('r', task.lock()->gpto_type(), rit->task->gpto_type(), event_id));
      event_id++;
      local_event_count++;
    }
    mock_idx -= rit->task->recv_events().size() * 2;
  }
}

void SetGPTOMode() {
  gpto_mode = kCompComm;  // default mode

  if (common::GetEnv("MS_ENABLE_GPTO_MODE") != "") {
    gpto_mode = static_cast<GPTO_MODE>(stoll(common::GetEnv("MS_ENABLE_GPTO_MODE")));
    if (gpto_mode >= kNumModes) {
      MS_LOG(WARNING) << "Wrong mode input, exiting GPTO...";
      return;
    }
  } else if (common::IsDisableRuntimeConfig(common::kRuntimeMultiStream)) {
    gpto_mode = kComp;
  } else if (common::IsEnableRuntimeConfig(common::kRuntimeMultiStream)) {
    gpto_mode = kCompComm;
  } else {
    gpto_mode = kCompCommGroup;
  }
  MS_LOG(INFO) << "Gpto mode value: " << gpto_mode;
}

void UpdateExecutionOrder(const KernelGraphPtr &kernel_graph, const SchedulingOutput &scheduling_output) {
  std::vector<Interval> task_times = scheduling_output.task_times;
  std::sort(task_times.begin(), task_times.end(), [](const Interval &x, const Interval &y) {
    return x.start < y.start || (x.start == y.start && x.task->gpto_type() > y.task->gpto_type());
  });
  std::vector<CNodePtr> new_exec_order;
  new_exec_order.reserve(task_times.size());
  for (size_t j = 0; j < task_times.size(); j++) {
    new_exec_order.push_back(task_times[j].task->cnode());
  }
  kernel_graph->set_execution_order(new_exec_order);
}

void GPTO(mindspore::device::DeviceResManager *device_res_manager, const KernelGraphPtr &kernel_graph,
          std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> *mock_exec_order) {
  if (kernel_graph->is_from_single_op() || kernel_graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "GPTO is not used when pynative forward or can't parse graph with dynamic shape now.";
    return;
  }

  std::pair<Time, Memory> baseline;

  SetGPTOMode();

  MS_LOG(INFO) << "Start Scheduling Subgraph " << kernel_graph << " with id " << kernel_graph->graph_id()
               << " and Execution order size " << kernel_graph->execution_order().size();

  std::unordered_map<CNodePtr, GptoTaskPtr> cnode_to_task;
  std::map<std::string, size_t> group_to_id;
  MS_LOG(INFO) << "Start ExtractSchedulingInput";
  PARAMETER_SIZE = 0;
  SchedulingInput scheduling_input =
    ExtractSchedulingInput(device_res_manager, kernel_graph, &cnode_to_task, &group_to_id);
  MS_LOG(INFO) << "Parameter size: " << PARAMETER_SIZE;
  MS_LOG(INFO) << "End ExtractSchedulingInput";

  if (scheduling_input.tasks.size() == 0) {
    MS_LOG(WARNING) << "Scheduling input doesn't have any tasks: skipping";
    return;
  }

  MS_LOG(INFO) << "Start Graph Output Process";
  GraphOutputProcess(kernel_graph,
                     &cnode_to_task);  //  kGraphOutput: "semilifelong end" functionality by default in memory
                                       //  estimated in source's memory impact and never "deallocated"
  MS_LOG(INFO) << "End Graph Output Process";

  MS_LOG(INFO) << "Start Update Ref Node for Gpto";
  UpdateRefNodeGpto(kernel_graph, &cnode_to_task);
  MS_LOG(INFO) << "End Update Ref Node for Gpto";

  Memory memory_lower_bound = 0;
  std::set<GptoTensorPtr, GptoTensorIdSort> tensors;
  auto can_debug = GetDebugConfig();
  if (can_debug.first) {
    // Memory lower bound (optional: for analysis only)
    std::vector<mindspore::somas::DynamicBitSet> nodes_dependency;

    MS_LOG(INFO) << "Start Compute Ancestors Descendants";
    ComputeAncestorsDescendants(scheduling_input.tasks, &nodes_dependency);
    MS_LOG(INFO) << "End Compute Ancestors Descendants";

    MS_LOG(INFO) << "Start Memory Lower Bound";
    ExtractTensors(scheduling_input.tasks, &tensors);
    memory_lower_bound = MemoryLowerBound(scheduling_input.tasks, nodes_dependency, tensors);
    MS_LOG(INFO) << "End Memory Lower Bound";

    // Baseline log
    MS_LOG(INFO) << "Start Baseline Greedy Scheduling";
    baseline = LogBaseline(&cnode_to_task, kernel_graph, can_debug.second, true);
    MS_LOG(INFO) << "End Baseline Greedy Scheduling";

    const size_t kPrecision = 5;
    MS_LOG(INFO) << "Speedup of baseline solution is " << std::setprecision(kPrecision)
                 << 1.0 * TotalTime(scheduling_input.tasks) / baseline.first;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CellReuseLevel() == CellReuseLevel::kNoInline) {
    if (can_debug.first) {
      MEMORY_LIMIT = baseline.second;
    } else {
      MEMORY_LIMIT = LogBaseline(&cnode_to_task, kernel_graph, can_debug.second, false).second;
    }
    const float memory_relax_no_inline = 1.02;
    MEMORY_LIMIT = static_cast<Memory>(memory_relax_no_inline * MEMORY_LIMIT + PARAMETER_SIZE);
  } else {
    if (common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT") != "") {
      MEMORY_LIMIT = static_cast<Memory>(stof(common::GetEnv("MS_ENABLE_GPTO_MEMORY_LIMIT")) * kGBToByte);
    } else {
      const float memory_safety = 0.98;
      MEMORY_LIMIT =
        static_cast<Memory>(context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY) * kGBToByte * memory_safety);
    }
  }
  MS_LOG(INFO) << "Memory Limit value: " << MEMORY_LIMIT;

  MS_LOG(INFO) << "Start GPTO Process";
  auto scheduling_output = MemAwareScheduler(scheduling_input, &group_to_id);
  MS_LOG(INFO) << "End GPTO Process";

  if (scheduling_output.makespan == SIZE_MAX) {
    MS_LOG(INFO) << "Hard memory limit is not satisfied by any solution's memory estimate, exiting GPTO...";
    return;
  }

  if (common::GetEnv("MS_ENABLE_OFFLINE_GPTO") != "1") {
    // If online mode, generate appropriate events to impose schedule, memory safety, and execution order
    if (common::GetEnv("MS_ENABLE_GPTO_NO_COST") == "1") {
      // Update execution order and trigger default data dependency events
      UpdateExecutionOrder(kernel_graph, scheduling_output);
    } else {
      // Explicitly generate events and generate mock execution order including them
      size_t num_schedule_events = ScheduleToEvents(scheduling_output);
      size_t num_memory_events = 0;
      if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() == kOptimizeO0) {
        num_memory_events = MemorySafetyEvents(scheduling_output);
      }
      MockExecutionOrder(scheduling_output, mock_exec_order, num_schedule_events + num_memory_events);
    }
  }

  if (can_debug.first) {
    if (common::GetEnv("MS_ENABLE_OFFLINE_GPTO") != "1") {
      MS_LOG(INFO) << "Start GPTO PrintGraphExecuteOrder";
      kernel_graph->PrintGraphExecuteOrder();
      MS_LOG(INFO) << "End GPTO PrintGraphExecuteOrder";
    }

    // GPTO log
    MS_LOG(INFO) << "Start printing output log file";
    LogSchedulingOutput(scheduling_input, scheduling_output, cnode_to_task, kernel_graph, tensors, memory_lower_bound,
                        &group_to_id, can_debug.second);
    MS_LOG(INFO) << "End printing output log file";
  }
}
}  // namespace gpto
}  // namespace mindspore
