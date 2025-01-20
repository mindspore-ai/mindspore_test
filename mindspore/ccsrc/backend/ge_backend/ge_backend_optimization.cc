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

#include "backend/ge_backend/ge_backend_optimization.h"

#include <memory>
#include <string>
#include <set>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/optimizer/graph_optimizer.h"
// #include "backend/ge_backend/pass/scalar_ops_output_unify_mindir.h"
// #include "backend/ge_backend/pass/shape_unify_mindir.h"
// #include "backend/ge_backend/pass/maketuple_unify_mindir.h"
// #include "backend/ge_backend/pass/inputs_unify_mindir.h"
// #include "backend/ge_backend/pass/scalar_unify_mindir.h"
// #include "backend/ge_backend/pass/tuple_unify_mindir.h"

#include "backend/common/pass/erase_invalid_micro_depend.h"
#include "backend/common/pass/erase_not_cut_attr.h"
#include "backend/common/pass/switch_not_cut.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace opt {
namespace {
void DoUnifyMindIRPass(const FuncGraphPtr &graph, const std::shared_ptr<mindspore::opt::GraphOptimizer> &optimizer) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(INFO) << "Do unify mindir pass for graph " << graph->ToString();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_before_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  (void)optimizer->Optimize(graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_end_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

bool HasSwitchNode(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  const auto &nodes = TopoSort(func_graph->get_return());
  return std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    return node != nullptr && node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch);
  });
}

// Check if src_node depends on dst_node.
bool IsTopoDependNode(const std::set<AnfNodePtr> &checked_calls, const AnfNodePtr &node,
                      std::set<AnfNodePtr> *checked_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_node);
  if (checked_calls.find(node) != checked_calls.end()) {
    return true;
  }
  if (!node->isa<CNode>() || checked_node->find(node) != checked_node->end()) {
    return false;
  }

  (void)checked_node->emplace(node);
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (IsTopoDependNode(checked_calls, input, checked_node)) {
      return true;
    }
  }
  return false;
}

bool HasParallelSwitchCall(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> switch_calls;
  const auto &nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (!common::AnfAlgo::IsCallNode(node)) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->size() == 0 || cnode->input(0) == nullptr ||
        (!common::AnfAlgo::CheckPrimitiveType(cnode->input(0), prim::kPrimSwitch))) {
      continue;
    }
    switch_calls.emplace_back(node);
  }
  if (switch_calls.size() <= 1) {
    return false;
  }
  constexpr size_t kMaxSwitchInlineSize = 10;
  if (switch_calls.size() >= kMaxSwitchInlineSize) {
    MS_LOG(INFO) << "Disable switch inline for switch node:" << switch_calls.size() << " more than 10.";
    return true;
  }
  std::set<AnfNodePtr> checked_calls{switch_calls.front()};
  for (size_t i = 1; i < switch_calls.size(); ++i) {
    std::set<AnfNodePtr> checked_nodes;
    if (!IsTopoDependNode(checked_calls, switch_calls[i], &checked_nodes)) {
      MS_LOG(INFO) << "Switch call node:" << switch_calls[i]->DebugString() << " has other parallel call node.";
      return true;
    }
    checked_calls.emplace(switch_calls[i]);
  }
  return false;
}

bool IsFuncGraphSupportSwitchInline(const FuncGraphPtr &graph) {
  return HasParallelSwitchCall(graph) ||
         std::any_of(graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(),
                     [](const auto &sub_graph) { return sub_graph != nullptr && HasParallelSwitchCall(sub_graph); });
}

bool HasAbstractRefOutput(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->dynamic_len()) {
      return false;
    }
    if (std::any_of(seq_abs->elements().begin(), seq_abs->elements().end(),
                    [](const abstract::AbstractBasePtr &sub_abs) { return HasAbstractRefOutput(sub_abs); })) {
      return true;
    }
  }
  if (abs->isa<abstract::AbstractRefTensor>()) {
    return true;
  }
  return false;
}

bool IsNodeValid(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  } else if (common::AnfAlgo::IsNodeOutputDynamicShape(node)) {
    MS_LOG(INFO) << "Disable switch inline for dynamic shape node:" << node->DebugString();
    return false;
  } else if (node->isa<CNode>() && common::AnfAlgo::IsTypeTransformOp(common::AnfAlgo::GetCNodeName(node))) {
    MS_LOG(INFO) << "Disable switch inline for backoff node:" << node->DebugString();
    return false;
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() <= 1 || cnode->input(1) == nullptr || !(IsValueNode<FuncGraph>(cnode->input(1)))) {
      return true;
    }
    const auto &func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
    MS_EXCEPTION_IF_NULL(func_graph);
    if (std::any_of(func_graph->parameters().begin(), func_graph->parameters().end(), [](const AnfNodePtr &para) {
          return para != nullptr && para->abstract() != nullptr &&
                 para->abstract()->isa<abstract::AbstractSequence>() &&
                 (para->abstract()->cast<abstract::AbstractSequencePtr>()->dynamic_len() ||
                  para->abstract()->cast<abstract::AbstractSequencePtr>()->size() > 1);
        })) {
      MS_LOG(INFO) << "Disable switch inline for tuple input in graph:" << func_graph->ToString()
                   << " for partial node:" << node->DebugString();
      return false;
    }
  } else if (common::AnfAlgo::IsCallNode(node) && HasAbstractRefOutput(node->abstract())) {
    return false;
  }
  return true;
}

bool IsEnableControlFlowInline(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (std::any_of(
        graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(), [](const auto &sub_graph) {
          return sub_graph != nullptr && sub_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE) && HasSwitchNode(sub_graph);
        })) {
    MS_LOG(INFO) << "Set reuse level from:" << context->CellReuseLevel() << " to:" << CellReuseLevel::kNoInline;
    context->SetCellReuseLevel(CellReuseLevel::kNoInline);
  }

  static const auto is_disable_switch_inline = common::IsDisableRuntimeConfig(common::kRuntimeSwitchInline);
  if (is_disable_switch_inline) {
    MS_LOG(INFO) << "Disable switch inline by runtime config.";
    return false;
  }

  MS_EXCEPTION_IF_NULL(graph);
  // Not support recursive.
  if (std::any_of(graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(),
                  [](const auto &sub_graph) { return sub_graph->recursive(); })) {
    MS_LOG(INFO) << "Disable switch inline for recursive.";
    return false;
  }

  if (context->CellReuseLevel() != CellReuseLevel::kLazyInline) {
    auto is_include_no_switch_call = [](const FuncGraphPtr &graph) {
      MS_EXCEPTION_IF_NULL(graph);
      const auto &nodes = TopoSort(graph->get_return());
      for (const auto &node : nodes) {
        MS_EXCEPTION_IF_NULL(node);
        if (common::AnfAlgo::IsCallNode(node)) {
          const auto &cnode = node->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(cnode);
          if (!common::AnfAlgo::CheckPrimitiveType(cnode->input(0), prim::kPrimSwitch)) {
            return true;
          }
        }
      }
      return false;
    };
    if (is_include_no_switch_call(graph)) {
      MS_LOG(INFO) << "Disable switch inline for unsupported call node.";
      return false;
    }
    if (std::any_of(graph->func_graphs_used_total().begin(), graph->func_graphs_used_total().end(),
                    is_include_no_switch_call)) {
      MS_LOG(INFO) << "Disable switch inline for unsupported call node.";
      return false;
    }
  }
  const auto &all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  if (std::any_of(all_nodes.begin(), all_nodes.end(), [](const AnfNodePtr &node) { return !IsNodeValid(node); })) {
    return false;
  }
  MS_LOG(INFO) << "Start check parallel switch call.";
  if (IsFuncGraphSupportSwitchInline(graph)) {
    MS_LOG(INFO) << "Disable switch inline for parallel switch call node.";
    return false;
  }
  MS_LOG(INFO) << "Enable switch inline.";
  return true;
}
}  // namespace

void UnifyMindIRPass(const FuncGraphPtr &func_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("unify_mindir_pm");
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::EraseInvalidMicroDepend>());
  if (common::AnfAlgo::IsDynamicGraph(func_graph)) {
    unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::EraseNotCutAttr>());
  }
  if (IsEnableControlFlowInline(func_graph)) {
    unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::SwitchNotCut>());
  }
  optimizer->AddPassManager(unify_mindir_pm);

  DoUnifyMindIRPass(func_graph, optimizer);
  const auto &sub_graphs = func_graph->manager()->func_graphs_used_total(func_graph);
  for (const auto &sub_graph : sub_graphs) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    DoUnifyMindIRPass(sub_graph, optimizer);
  }
  (void)profiler::CollectHostInfo("GE", "Unify MindIR Pass", "UnifyMindIRPass", start_time, profiler::GetClockSyscnt(),
                                  0);
}

// void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph) {
//   uint64_t start_time = profiler::GetClockSyscnt();
//   MS_EXCEPTION_IF_NULL(func_graph);
//   auto context_ptr = MsContext::GetInstance();
//   MS_EXCEPTION_IF_NULL(context_ptr);
// #ifdef ENABLE_DUMP_IR
//   if (context_ptr->CanDump(kIntroductory)) {
//     std::string file_name = "hwopt_d_before_ge_dynamic_shape_unify_mindir_graph.ir";
//     DumpIR(file_name, func_graph);
//     DumpIRProto(func_graph, "before_ge_dynamic_shape_unify_mindir_hwopt");
//   }
// #endif
//   auto dynamic_unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("ge_dynamic_unify_mindir_pm");
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarOpsOutputUnifyMindIR>());
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ShapeUnifyMindIR>());
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::MakeTupleUnifyMindIR>());
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::InputsUnifyMindIR>());
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarUnifyMindIR>());
//   dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::TupleUnifyMindIR>());
//   auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
//   optimizer->AddPassManager(dynamic_unify_mindir_pm);
//   (void)optimizer->Optimize(func_graph);
// #ifdef ENABLE_DUMP_IR
//   if (context_ptr->CanDump(kIntroductory)) {
//     std::string file_name = "hwopt_d_after_ge_dynamic_shape_unify_mindir_graph.ir";
//     DumpIR(file_name, func_graph);
//   }
// #endif
//   (void)profiler::CollectHostInfo("GE", "GE Dynamic Shape Unify MindIR", "GEBackend_Dynamic_UnifyMindIR", start_time,
//                                   profiler::GetClockSyscnt(), 0);
// }
}  // namespace opt
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
