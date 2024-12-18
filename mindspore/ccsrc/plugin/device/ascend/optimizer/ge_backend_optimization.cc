/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"

#include <memory>
#include <string>

#include "backend/common/pass/dropout_gen_mask_fusion.h"
#include "backend/common/pass/common_subexpression_elimination.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/concat_outputs_for_all_gather.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "backend/common/pass/graph_view_replace_pass.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "backend/common/pass/split_inputs_for_reduce_scatter.h"
#include "backend/common/pass/overlap_grad_reduce.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include "plugin/device/ascend/optimizer/enhancer/eliminate_maketuple_getitem.h"
#include "plugin/device/ascend/optimizer/format_type/deal_ref_output.h"
#include "plugin/device/ascend/optimizer/format_type/set_fracz_group_attr.h"
#include "plugin/device/ascend/optimizer/ge/expander_fallback.h"
#include "plugin/device/ascend/optimizer/ge/insert_identity.h"
#include "plugin/device/ascend/optimizer/ge/process_call_inline.h"
#include "plugin/device/ascend/optimizer/ge/process_partial_inline.h"
#include "plugin/device/ascend/optimizer/ge/convert_pad_v3_paddings.h"
#include "plugin/device/ascend/optimizer/heterogeneous/insert_move_to.h"
#include "plugin/device/ascend/optimizer/ir_fission/seed_adapter.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/shape_reshape_fusion.h"
#include "plugin/device/ascend/optimizer/ge/hcom/insert_tensor_move_for_hccl_op_ge.h"
#include "plugin/device/ascend/optimizer/ge/resize_bilinear_add_attr.h"
#include "backend/common/pass/custom_defined_depend.h"

namespace mindspore {
namespace opt {
void AclAfterCreateKernel(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel create. graph id: "
                << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 0);
  PROF_START(AclAfterCreateKernel);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_acl_graph_final_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_ack = std::make_shared<PassManager>("opt_acl_ack");
  opt_acl_ack->AddPass(std::make_shared<EraseVisitAttr>());
  opt_acl_ack->AddPass(std::make_shared<DealRefOutput>());
  optimizer->AddPassManager(opt_acl_ack);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AclAfterCreateKernel);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_acl_graph_final_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 1);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass after kernel create. graph id: "
                << kernel_graph->graph_id();
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
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_pm = std::make_shared<PassManager>("opt_acl_pm");
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessCallInline>());
  opt_acl_pm->AddPass(std::make_shared<SeedAdapter>());

  if (common::IsEnableRuntimeConfig(common::kRuntimeInsertTensorMove)) {
    opt_acl_pm->AddPass(std::make_shared<opt::InsertTensorMoveForHcclOpGe>());
  } else {
    opt_acl_pm->AddPass(std::make_shared<InsertTensorMoveForCommunication>());
  }
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessPartialInline>());
  opt_acl_pm->AddPass(std::make_shared<opt::ExpanderFallback>());
  opt_acl_pm->AddPass(std::make_shared<opt::ConvertPadV3Paddings>());
  opt_acl_pm->AddPass(std::make_shared<opt::ConvertPadV3GradPaddings>());
  opt_acl_pm->AddPass(std::make_shared<opt::ResizeBilinearAddAttr>());
  opt_acl_pm->AddPass(std::make_shared<opt::CustomDefinedDepend>(false, kernel_graph->graph_id()));
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

void GEBackendOptimizeACLAfterKernelSelect(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel select. graph id: "
                << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(GEBackendOptimizeACLAfterKernelSelect);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_opt_acl_graph_after_kernel_select_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_after_kernel_select_pm = std::make_shared<PassManager>("opt_acl_after_kernel_select_pm");
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<SetFraczGroupAttr>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<InsertIdentity>());

  int execution_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  // graph_mode process the pass in OptimizeACLGraphAfterCreateKernel
  if (execution_mode == kPynativeMode) {
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<EraseVisitAttr>());
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<DealRefOutput>());
  }

  if (!kernel_graph->is_from_single_op() && !kernel_graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  }
  if (!kernel_graph->is_graph_run_mode() && context_ptr->ascend_soc_version() != "ascend910") {
    opt_acl_after_kernel_select_pm->AddFusionPass(std::make_shared<opt::ShapeReshapeFusion>());
    opt_acl_after_kernel_select_pm->AddFusionPass(std::make_shared<opt::ShapeReshapeDirectFusion>());
  }
  optimizer->AddPassManager(opt_acl_after_kernel_select_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEBackendOptimizeACLAfterKernelSelect);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_end_opt_acl_graph_after_kernel_select_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelSelect",
                                  start_time, profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void GEBackendOptimizeACLAfterKernelPacket(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (common::IsDisableRuntimeConfig(common::kRuntimeView) || context_ptr->IsEnableInferBoost() ||
      kernel_graph->is_from_single_op()) {
    return;
  }

  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel packet. graph id: "
                << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(GEBackendOptimizeACLAfterKernelPacket);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_opt_acl_graph_after_kernel_packet_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_after_kernel_packet_pm = std::make_shared<PassManager>("opt_acl_after_kernel_packet");
  opt_acl_after_kernel_packet_pm->AddPass(std::make_shared<opt::GraphViewReplacePass>());
  PROF_END(GEBackendOptimizeACLAfterKernelPacket);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_end_opt_acl_graph_after_kernel_packet_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  optimizer->AddPassManager(opt_acl_after_kernel_packet_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelPacket",
                                  start_time, profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void GEUnifyMindIR(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(GEUnifyMindIR);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
    DumpIRProto(kernel_graph, "before_unify_mindir_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(GetBackendCommonUnifyMindIRPassManager());
  optimizer->AddPassManager(GetBackendFusionGroupPassManager());
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEUnifyMindIR);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  (void)profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_UnifyMindIR", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void GEAfterInlineOptimize(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(GEAfterInlineOptimize);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_inline_optimize_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto after_inline_pm = std::make_shared<PassManager>("after_inline_pm");
  after_inline_pm->AddFusionPass(std::make_shared<DropoutGenMaskFusion>());
  after_inline_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  after_inline_pm->AddPass(std::make_shared<EliminateMaketupleGetitem>());
  after_inline_pm->AddPass(std::make_shared<InsertMoveTo>());
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_GRAD_COMM_OPT)) {
    after_inline_pm->AddPass(std::make_shared<OverlapGradReduce>());
  }
  optimizer->AddPassManager(after_inline_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEAfterInlineOptimize);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_inline_optimize_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  (void)profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_AfterInline", start_time,
                                  profiler::GetClockSyscnt(), 0);
}
}  // namespace opt
}  // namespace mindspore
