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

#include "backend/ge_backend/pass/ge_backend_optimization.h"

#include <memory>
#include <string>
#include <set>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "backend/ge_backend/pass/scalar_ops_output_unify_mindir.h"
#include "backend/ge_backend/pass/shape_unify_mindir.h"
#include "backend/ge_backend/pass/maketuple_unify_mindir.h"
#include "backend/ge_backend/pass/inputs_unify_mindir.h"
#include "backend/ge_backend/pass/scalar_unify_mindir.h"
#include "backend/ge_backend/pass/tuple_unify_mindir.h"

#include "backend/common/pass/erase_invalid_micro_depend.h"
#include "backend/common/pass/erase_not_cut_attr.h"
#include "backend/common/pass/switch_not_cut.h"

#include "backend/ge_backend/pass/histogram_fixed_width_fusion.h"
#include "backend/ge_backend/pass/tensor_array.h"
#include "backend/ge_backend/pass/specialized_prepare.h"
#include "backend/ge_backend/pass/centralization_mindir.h"
#include "backend/ge_backend/pass/trans_depend_value_to_int32.h"
#include "backend/ge_backend/pass/add_parallel_group_id_attr.h"
#include "backend/ge_backend/pass/ge_convert_const_input_to_tensor_input.h"
#include "backend/ge_backend/pass/remove_tensor_to_scalar_or_tuple_ops.h"
#include "backend/ge_backend/pass/all_to_all_v_for_ge.h"
#include "backend/ge_backend/pass/fa_alltoallv_parallel.h"
#include "backend/ge_backend/pass/hcom/insert_load_for_allgather.h"
#include "backend/ge_backend/pass/hcom/insert_depend_for_all_gather_ge.h"
#include "backend/ge_backend/pass/convert_condition_input_to_scalar.h"
#include "backend/ge_backend/pass/convert_data_depend_to_control_depend.h"
#include "backend/ge_backend/pass/maketuple_depend_remover.h"
#include "backend/ge_backend/pass/fused_cast_add.h"
#include "backend/ge_backend/pass/hcom/add_parallel_group_for_hcom.h"
#include "backend/ge_backend/pass/expand_dims_for_batchnorm.h"
#include "backend/ge_backend/pass/dropout_gen_mask_depend.h"
#include "backend/ge_backend/pass/add_cast_for_ge.h"
#include "backend/ge_backend/pass/nan_to_num_for_ge.h"
#include "backend/ge_backend/pass/unfold_nested_output.h"
#include "backend/ge_backend/pass/unfold_maketuple.h"
#include "backend/ge_backend/pass/broadcast_for_select.h"
#include "backend/ge_backend/pass/add_noop_to_es_grad.h"
#include "backend/ge_backend/pass/bce_with_logits_loss_for_ge.h"
#include "backend/common/pass/custom_defined_depend.h"
// todo: this passes mv to backend_common from plugin
#include "plugin/device/ascend/optimizer/ge/hcom/insert_tensor_move_for_hccl_op_ge.h"
#include "plugin/device/ascend/optimizer/ge/resize_bilinear_add_attr.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "plugin/device/ascend/optimizer/ge/process_call_inline.h"
#include "plugin/device/ascend/optimizer/ir_fission/seed_adapter.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "plugin/device/ascend/optimizer/ge/process_partial_inline.h"
#include "plugin/device/ascend/optimizer/ge/expander_fallback.h"
#include "plugin/device/ascend/optimizer/ge/convert_pad_v3_paddings.h"
#include "plugin/device/ascend/optimizer/ge/convert_embedding_dense_grad_padding.h"

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

void MarkRefGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Mark graph is ref graph: " << kernel_graph->graph_id();
  auto manager = kernel_graph->manager();
  if (manager == nullptr || kernel_graph->has_attr(kIsRefGraph)) {
    return;
  }
  for (const auto &node : TopoSort(kernel_graph->get_return(), SuccDeeperSimple, AlwaysInclude)) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, cnode) &&
                          common::AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
    if (!(is_side_effect && cnode->fullname_with_scope().find("optimizer") != std::string::npos)) {
      continue;
    }
    for (const auto &node_pair : manager->node_users()[cnode]) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimUpdateState)) {
        kernel_graph->set_attr(kIsRefGraph, MakeValue(true));
        MS_LOG(INFO) << "graph is ref graph: " << kernel_graph->graph_id();
        return;
      }
    }
  }
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

void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_ge_dynamic_shape_unify_mindir_graph.ir";
    DumpIR(file_name, func_graph);
    DumpIRProto(func_graph, "before_ge_dynamic_shape_unify_mindir_hwopt");
  }
#endif
  auto dynamic_unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("ge_dynamic_unify_mindir_pm");
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarOpsOutputUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ShapeUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::MakeTupleUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::InputsUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::TupleUnifyMindIR>());
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  optimizer->AddPassManager(dynamic_unify_mindir_pm);
  (void)optimizer->Optimize(func_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_ge_dynamic_shape_unify_mindir_graph.ir";
    DumpIR(file_name, func_graph);
  }
#endif
  (void)profiler::CollectHostInfo("GE", "GE Dynamic Shape Unify MindIR", "GEBackend_Dynamic_UnifyMindIR", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void GEBackendOptimization(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
  PROF_START(GEBackendOptimization);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto opt_ge_pm = std::make_shared<mindspore::opt::PassManager>("opt_ge_pm");
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::HistogramFixedWidthFusion>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayAddFlowCond1>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayAddFlowCond2>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::GeTensorArrayCastIndex>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayPrepare>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::CentralizationMindIR>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TransDependValueToInt32>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddParallelGroupIdAttr>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::GEConvertConstInputToTensorInput>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::RemoveTensorToScalarOrTupleOps>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AlltoAllVForGE>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::FaAlltoAllvParallel>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertLoadForAllGather>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForHcclOpGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertDependForAllGatherGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ConvertCondInputToScalar>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ConvertDataDependToControlDepend>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::MakeTupleDependRemover>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::FusedCastAdd>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddParallelGroupForHcom>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ExpandDimsForBatchNorm>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::DropoutGenMaskDepend>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddCastForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::NanToNumForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ResizeBilinearAddAttr>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AscendConvertTupleInputToDynamicInput>(true, true));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::UnfoldNestedOutput>("unfold_nested_output"));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::UnfoldMaketuple>("unfold_nested_maketuple"));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::BroadCastForSelect>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddNoOpToESGrad>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::BCEWithLogitsLossForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::CustomDefinedDepend>(true, kernel_graph->graph_id()));

  optimizer->AddPassManager(opt_ge_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  PROF_END(GEBackendOptimization);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
}

void GEBackendOptimizeACL(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(GEBackendOptimizeACL);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto opt_acl_pm = std::make_shared<mindspore::opt::PassManager>("opt_acl_pm");
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ProcessCallInline>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::SeedAdapter>());

  if (common::IsEnableRuntimeConfig(common::kRuntimeInsertTensorMove)) {
    opt_acl_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForHcclOpGe>());
  } else {
    opt_acl_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForCommunication>());
  }
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ProcessPartialInline>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ExpanderFallback>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertPadV3Paddings>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertPadV3GradPaddings>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertEmbeddingDenseGradPadding>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ResizeBilinearAddAttr>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::CustomDefinedDepend>(false, kernel_graph->graph_id()));
  optimizer->AddPassManager(opt_acl_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEBackendOptimizeACL);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", start_time,
                                  profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void OptimizeGEGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(OptimizeGEGraph);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize ge graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
  }
  MarkRefGraph(graph);
  GEBackendOptimizeACL(graph);
  GEBackendOptimization(graph);

  for (auto &child_graph : graph->child_graph_order()) {
    if (child_graph.lock()->has_flag(kFlagGeKernel)) {
      continue;
    }
    OptimizeGEGraph(child_graph.lock(), memo);
  }
  PROF_END(OptimizeGEGraph);
  MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
}
}  // namespace opt
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
