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
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include <string>
#include <utility>
#include <algorithm>
#include <deque>
#include "include/common/utils/parallel_context.h"
#include "debug/profiler/profiler.h"
#include "common/kernel.h"
#include "mindapi/base/type_id.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "backend/common/session/kernel_graph_mgr.h"
#include "plugin/res_manager/ascend/device_context_conf/op_debug_conf.h"
#include "plugin/res_manager/ascend/device_context_conf/op_precision_conf.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/hal/hardware/acl_somas.h"
#include "plugin/device/ascend/hal/hardware/acl_stream_assign.h"
#include "plugin/device/ascend/hal/hardware/gpto.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_build.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"
#include "plugin/device/ascend/kernel/simu/simu_kernel_build.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "kernel/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"
#include "plugin/device/ascend/kernel/graph_kernel/kernel_packet_ascend_kernel_mod.h"
#include "plugin/res_manager/ascend/mbuf_manager/tensorreport_utils.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#ifdef ENABLE_DVM
#include "plugin/device/ascend/kernel/dvm/dvm_kernel_build.h"
#endif

#include "mindspore/ops/kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/hardware/device_context_manager.h"
#include "pyboost/pyboost_utils.h"
#include "pyboost/op_runner.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "kernel/ascend/opapi/aclnn_kernel_build.h"
#include "kernel/ascend/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/atb/atb_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "common/ms_factory.h"
#include "common/kernel_build_info.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "kernel/ascend/acl_ir/ge_adapter_info.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_compiler_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "debug/profiler/profiling.h"
#include "utils/anf_utils.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "kernel/ascend/availability/silent_check/ascend_silent_check.h"
#include "kernel/ascend/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::device::ascend {
namespace {
constexpr size_t kSwitchInputSize = 3;
constexpr size_t kSwitchCondIndex = 1;
constexpr size_t kSwitchBranchTrueIndex = 2;
constexpr size_t kSwitchBranchFalseIndex = 3;

std::string GetKernelTypeStr(const KernelType &kernel_type) {
  std::string type = "";
  if (kernel_type == KernelType::ACL_KERNEL) {
    type = "acl_kernel";
  } else if (kernel_type == KernelType::HOST_KERNEL) {
    type = "host_kernel";
  } else if (kernel_type == KernelType::HCCL_KERNEL) {
    type = "hccl_kernel";
  } else if (kernel_type == KernelType::OPAPI_KERNEL) {
    type = "opapi_kernel";
  } else if (kernel_type == KernelType::INTERNAL_KERNEL) {
    type = "internal_kernel";
  } else if (kernel_type == KernelType::AKG_KERNEL) {
    type = "akg_kernel";
  } else if (kernel_type == KernelType::RT_KERNEL) {
    type = "rt_kernel";
  } else if (kernel_type == KernelType::ATB_KERNEL) {
    type = "atb_kernel";
  }
  return type;
}

kernel::KernelModPtr GenerateAkgKernelMod(const CNodePtr &kernel);

void RegisterSilentCheckForNode(const CNodePtr &kernel, const kernel::KernelModPtr &kernel_mod_ptr,
                                const std::string &op_name, KernelType kernel_type) {
  if (silentcheck::ascend::SilentChecker::GetInstance().IsCommOpInputNotSupport()) {
    return;
  }
  if (!kernel->HasPrimalAttr(silentcheck::kAttrSilentCheckOpType)) {
    return;
  }
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
  if (input_kernel_tensors.empty() || input_kernel_tensors[0]->IsDynamicShape()) {
    return;
  }
  auto check_op_type = GetValue<int>(kernel->GetPrimalAttr(silentcheck::kAttrSilentCheckOpType));
  auto tensor_type = input_kernel_tensors[0]->dtype_id();
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Type of input0 of " << kernel->fullname_with_scope() << " is "
                                  << input_kernel_tensors[0]->GetType()->ToString();
  if (tensor_type != kNumberTypeFloat32 && tensor_type != kNumberTypeBFloat16) {
    if (check_op_type == silentcheck::kSilentCheckGradCommOp) {
      MS_LOG(WARNING)
        << "Input 0 of node " << kernel->fullname_with_scope() << " is " << input_kernel_tensors[0]->ToString()
        << ", silent check only support bfloat16 and float32 type, silent check function will be disabled.";
      silentcheck::ascend::SilentChecker::GetInstance().SetCommOpInputNotSupport(true);
      silentcheck::ascend::SilentChecker::GetInstance().ClearCheckHooks();
    } else {
      MS_LOG(WARNING) << "Input 0 of node " << kernel->fullname_with_scope() << " is "
                      << input_kernel_tensors[0]->ToString()
                      << ", silent check only support bfloat16 and float32 type, skip silent check for this node";
    }
    return;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Register silent check for kernel opname:" << op_name
                                  << ", kernel type:" << GetKernelTypeStr(kernel_type) << " "
                                  << kernel->fullname_with_scope();
  silentcheck::ascend::SilentChecker::GetInstance().RegisterCheck(kernel_mod_ptr, input_kernel_tensors[0]);
}

bool GenerateKernelMod(const std::vector<CNodePtr> &kernels) {
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelMod(kernel)) {
      continue;
    }
    if (AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    std::string opname = common::AnfAlgo::GetCNodeName(kernel);
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    auto kernel_type = AnfAlgo::GetKernelType(kernel);
    if (kernel_type == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (kernel_type == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else if (kernel_type == KernelType::HCCL_KERNEL) {
      if (common::IsExecuteSimulation()) {
        kernel_mod_ptr = kernel::SimuOpBuild(kernel);
      } else {
        kernel_mod_ptr = kernel::HcclOpBuild(kernel);
      }
    } else if (kernel_type == KernelType::OPAPI_KERNEL) {
      kernel_mod_ptr = kernel::AclnnOpBuild(kernel);
    } else if (kernel_type == KernelType::AKG_KERNEL) {
      kernel_mod_ptr = GenerateAkgKernelMod(kernel);
    } else if (kernel_type == KernelType::RT_KERNEL) {
      kernel_mod_ptr = kernel::RtOpBuild(kernel);
    } else if (kernel_type == KernelType::INTERNAL_KERNEL) {
      kernel_mod_ptr = kernel::InternalKernelBuild(kernel);
    } else if (kernel_type == KernelType::ATB_KERNEL) {
      kernel_mod_ptr = kernel::AtbKernelBuild(kernel);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, kernel)
        << "The kernel: " << kernel->fullname_with_scope()
        << " kernel build failed, kernel type: " << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_LOG(INFO) << "kernel opname:" << opname << ", kernel type:" << GetKernelTypeStr(kernel_type);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    RegisterSilentCheckForNode(kernel, kernel_mod_ptr, opname, kernel_type);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}

kernel::KernelModPtr CreateKernelPacketKernelMod(const CNodePtr &kernel) {
  MS_LOG(DEBUG) << "Build KernelPacket: " << kernel->DebugString();
  auto real_kernel = kernel::GetKernelPacketRealNode(kernel);
  if (!GenerateKernelMod({real_kernel})) {
    MS_LOG(ERROR) << "Build " << real_kernel->DebugString() << " failed.";
    return nullptr;
  }
  auto kp_kernelmod = std::make_shared<kernel::KernelPacketAscendKernelMod>();
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  // when the kernel is inlined, multiple kernels may share a Primitive, so clone an object.
  prim = prim->Clone();
  kernel->set_input(0, NewValueNode(prim));
  auto name = GetValue<std::string>(prim->GetAttr(kAttrKernelPacketNode));
  auto infer_func = std::make_shared<kernel::KernelPacketInfer>(name, real_kernel->func_graph(), kp_kernelmod.get());
  prim->set_attr("infer_shape_functor", infer_func);
  auto real_kernel_info = dynamic_cast<device::KernelInfo *>(real_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(real_kernel_info);
  if (!kp_kernelmod->KernelMod::Init(prim, input_kernel_tensors, output_kernel_tensors) ||
      !kernel::KernelPacketInitializer::InitKernel(real_kernel, real_kernel_info->GetKernelMod(), kp_kernelmod.get(),
                                                   infer_func.get())) {
    MS_LOG_WITH_NODE(EXCEPTION, real_kernel)
      << "#dmsg#Kernel build failed:#dmsg#Initialize kernel op[" << real_kernel->fullname_with_scope() << "] failed.";
  }
  real_kernel_info->set_kernel_mod(nullptr);
  return kp_kernelmod;
}

kernel::KernelModPtr GenerateAkgKernelMod(const CNodePtr &kernel) {
  if (common::AnfAlgo::HasNodeAttr(kAttrKernelPacketNode, kernel)) {
    return CreateKernelPacketKernelMod(kernel);
  }
#ifdef ENABLE_DVM
  return kernel::DvmOpBuild(kernel);
#else
  return nullptr;
#endif
}

void SetAclOpPrecisionMode() {
  auto op_precision_conf = OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);

  auto precision_mode = op_precision_conf->precision_mode();
  if (precision_mode.empty()) {
    precision_mode =
      (device::ascend::AclUtil::KeepOriginDType() == 1) ? "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  }
  MS_LOG(INFO) << "Set aclop PRECISION_MODE: " << precision_mode;
  auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_PRECISION_MODE, precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set precision mode failed! Error flag is " << ret;
  }

  auto op_precision_mode = op_precision_conf->op_precision_mode();
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(INFO) << "Set aclop OP_PRECISION_MODE: " << op_precision_mode;
  ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! Error flag is " << ret;
  }
}

void SelectKernelInfo(const KernelGraphPtr &kernel_graph, const CNodePtr &kernel, bool *is_aclop = nullptr,
                      std::vector<size_t> *op_selected_num = nullptr) {
  auto [select_res, msg, etype, is_aclop_local] =
    device::ascend::SelectKernelInfoWithMsg(kernel_graph, kernel, op_selected_num);
  if (is_aclop != nullptr) {
    *is_aclop = is_aclop_local;
  }
  if (!select_res) {
    MS_LOG(INFO) << "node is " << kernel->fullname_with_scope() << " should backoff";
    std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
    device::ascend::HandleKernelSelectFailure(kernel_graph, kernel, failure_info);
  }
}

void SelectKernel(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *const memo,
                  std::vector<size_t> *op_selected_num, bool *has_aclop = nullptr) {
  // select kernel
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(SelectKernel);
  if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  memo->insert(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    bool is_aclop = false;
    SelectKernelInfo(kernel_graph, kernel, &is_aclop, op_selected_num);
    if (has_aclop != nullptr) {
      // cppcheck-suppress useStlAlgorithm
      *has_aclop |= is_aclop;
    }
  }
  if (!kernel_graph->is_from_single_op()) {
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
  }
  for (auto &child_graph : kernel_graph->child_graph_order()) {
    SelectKernel(child_graph.lock(), memo, op_selected_num);
  }
  PROF_END(SelectKernel);
}

inline void PrintOpSelectedNum(const std::vector<size_t> &op_selected_num) {
  MS_LOG(INFO) << "Number of INTERNAL_KERNEL, OPAPI_KERNEL, ACL_KERNEL, ATB_KERNEL, HCCL_KERNEL, HOST_KERNEL:";
  MS_VLOG(VL_FLOW) << "Number of INTERNAL_KERNEL, OPAPI_KERNEL, ACL_KERNEL, ATB_KERNEL, HCCL_KERNEL, HOST_KERNEL:";
  std::stringstream ss;
  for (const auto num : op_selected_num) {
    ss << num << " ";
  }
  MS_LOG(INFO) << ss.str();
  MS_VLOG(VL_FLOW) << ss.str();
}

void InlineSubGraph(const KernelGraphPtr &graph, const KernelGraphPtr &sub_graph, CNodePtr kernel_cnode,
                    AnfNodePtr *last_call, AnfNodePtr *pre_last_call, bool is_switch_inline) {
  MS_EXCEPTION_IF_NULL(kernel_cnode);
  MS_EXCEPTION_IF_NULL(sub_graph);
  MS_LOG(INFO) << "InlineSubGraph: " << kernel_cnode->fullname_with_scope() << ", sub graph: " << sub_graph->graph_id()
               << ", need inline: " << sub_graph->need_inline();
  auto main_graph = kernel_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(main_graph);
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = MakeManager({main_graph});
    MS_EXCEPTION_IF_NULL(mng);
    mng->AddFuncGraph(main_graph);
    main_graph->set_manager(mng);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_cnode->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  AnfNodePtrList inp;
  auto &call_input = kernel_cnode->inputs();
  // let operators on different subgraphs will not be executed interleavedly
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto pp_1f1b_value = ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  for (size_t i = 1; i < call_input.size(); i++) {
    if ((pp_1f1b_value.empty() || !kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp)) && last_call != nullptr &&
        (*last_call) != nullptr) {
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), call_input[i], (*last_call)});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(call_input[i]->abstract());
      depend->AddAttr("inline_depend", MakeValue(true));
      if (!pp_1f1b_value.empty() && pre_last_call != nullptr && (*pre_last_call) != nullptr) {
        auto depend_new = graph->NewCNode({NewValueNode(prim::kPrimDepend), depend, (*pre_last_call)});
        depend_new->set_abstract(call_input[i]->abstract());
        depend_new->AddAttr("inline_depend2", MakeValue(true));
        inp.push_back(depend_new);
      } else {
        inp.push_back(depend);
      }
    } else if (kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp) && pre_last_call != nullptr &&
               (*pre_last_call) != nullptr) {
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), call_input[i], (*pre_last_call)});
      depend->set_abstract(call_input[i]->abstract());
      depend->AddAttr("inline_depend1", MakeValue(true));
      inp.push_back(depend);
    } else {
      inp.push_back(call_input[i]);
    }
  }
  if (pre_last_call != nullptr && last_call != nullptr) {
    (*pre_last_call) = (*last_call);
  }
  const auto &ref_map = sub_graph->GetRefMap();
  auto out = session::KernelGraphMgr::DoInline(sub_graph, main_graph, inp, kernel_cnode, kernel_info->graph_id(),
                                               ref_map, graph, is_switch_inline);
  (void)mng->Replace(kernel_cnode, out);
  // Inline graph boundary: MakeTuple---->Depend---->Tensormove
  // Avoid long link times at runtime
  if (last_call != nullptr) {
    static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
    if (!enable_infer_boost) {
      auto value_node = graph->NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(1)));
      MS_EXCEPTION_IF_NULL(value_node);
      CNodePtr tensor_move = nullptr;
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), value_node, out});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(value_node->abstract());
      tensor_move = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTensorMove->name())), depend});
      MS_EXCEPTION_IF_NULL(tensor_move);
      tensor_move->set_abstract(value_node->abstract());
      common::AnfAlgo::SetNodeAttr(kAttrKernelGraphBoundary, MakeValue(sub_graph->ToString()), tensor_move);
      // select kernel
      SelectKernelInfo(graph, tensor_move);
      (*last_call) = tensor_move;
    } else {
      (*last_call) = out;
    }
  }
}

void InlineCallGraph(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  PROF_START(InlineCallGraph);
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  graph->SetExecOrderByDefault();
  auto kernel_cnodes = graph->execution_order();
  AnfNodePtr last_call = nullptr;
  AnfNodePtr pre_last_call = nullptr;
  std::vector<FuncGraphManagerPtr> subgraph_managers;
  for (auto &kernel_cnode : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimCallInline)) {
      auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      auto mng = inline_subgraph->manager();
      if (mng == nullptr) {
        auto manager = MakeManager({inline_subgraph}, false);
        MS_EXCEPTION_IF_NULL(manager);
        manager->AddFuncGraph(inline_subgraph);
        inline_subgraph->set_manager(manager);
        // subgraph is not changed when InlineSubGraph, hold the manager of subgraph to avoid being released.
        subgraph_managers.emplace_back(manager);
      }
      InlineSubGraph(graph, inline_subgraph, kernel_cnode, &last_call, &pre_last_call, false);
    }
  }
  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterInline(graph);
  PROF_END(InlineCallGraph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

CNodePtr GetCondSwitchNode(const KernelGraphPtr &graph, const std::map<AnfNodePtr, size_t> &branch_input,
                           const AnfNodePtr &cond, std::map<AnfNodePtr, AnfNodePtr> *branch_tuple_getitem) {
  std::vector<AnfNodePtr> cond_switch_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionSwitch->name()))};
  cond_switch_inputs.resize(branch_input.size() + kIndex2);
  cond_switch_inputs[kIndex1] = cond;
  for (auto &kv : branch_input) {
    cond_switch_inputs[kv.second + kIndex2] = kv.first;
  }
  auto cond_switch_node = graph->NewCNode(cond_switch_inputs);
  MS_EXCEPTION_IF_NULL(cond_switch_node);

  for (auto &kv : branch_input) {
    if (branch_tuple_getitem->find(kv.first) == branch_tuple_getitem->end()) {
      auto tuple_getitem_node =
        graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cond_switch_node, NewValueNode(SizeToLong(kv.second))});
      MS_EXCEPTION_IF_NULL(tuple_getitem_node);
      tuple_getitem_node->set_abstract(kv.first->abstract());
      (*branch_tuple_getitem)[kv.first] = tuple_getitem_node;
    }
  }
  AbstractBasePtrList abstract_list;
  for (size_t i = kIndex2; i < cond_switch_inputs.size(); ++i) {
    (void)abstract_list.emplace_back(cond_switch_inputs[i]->abstract());
  }
  cond_switch_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  SelectKernelInfo(graph, cond_switch_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(cond_switch_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  for (size_t input_index = 1; input_index < common::AnfAlgo::GetInputNum(cond_switch_node); ++input_index) {
    kernel_info->AddRefMap(input_index - 1, input_index);
  }
  return cond_switch_node;
}

CNodePtr GetBranchNode(const KernelGraphPtr &graph, const CNodePtr &old_branch_node,
                       const std::map<AnfNodePtr, AnfNodePtr> &branch_tuple_getitem) {
  std::vector<AnfNodePtr> branch_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartialInline->name()))};
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(old_branch_node); i++) {
    auto input = common::AnfAlgo::GetInputNode(old_branch_node, i);
    if (branch_tuple_getitem.find(input) != branch_tuple_getitem.end()) {
      branch_inputs.push_back(branch_tuple_getitem.at(input));
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, old_branch_node)
        << "Invalid input of branch node: " << old_branch_node->fullname_with_scope() << ", " << i << ", "
        << input->fullname_with_scope();
    }
  }
  auto branch_node = graph->NewCNode(branch_inputs);
  MS_EXCEPTION_IF_NULL(branch_node);
  branch_node->set_abstract(old_branch_node->abstract());
  SelectKernelInfo(graph, branch_node);
  common::AnfAlgo::CopyNodeAttrs(old_branch_node, branch_node);
  return branch_node;
}

void CollectInputByBranchCNode(const CNodePtr &true_branch_cnode, const CNodePtr &false_branch_cnode,
                               std::map<AnfNodePtr, size_t> *branch_input) {
  MS_EXCEPTION_IF_NULL(true_branch_cnode);
  MS_EXCEPTION_IF_NULL(false_branch_cnode);
  MS_EXCEPTION_IF_NULL(branch_input);
  std::set<AnfNodePtr> monad_inputs;
  auto now_input_cnt = 0;
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(true_branch_cnode); i++) {
    auto input = common::AnfAlgo::GetInputNode(true_branch_cnode, i);
    if (branch_input->find(input) != branch_input->end() || monad_inputs.find(input) != monad_inputs.end()) {
      continue;
    }
    if (HasAbstractMonad(input)) {
      monad_inputs.emplace(input);
      continue;
    }
    (*branch_input)[input] = now_input_cnt++;
  }
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(false_branch_cnode); i++) {
    auto input = common::AnfAlgo::GetInputNode(false_branch_cnode, i);
    if (branch_input->find(input) != branch_input->end() || monad_inputs.find(input) != monad_inputs.end()) {
      continue;
    }
    if (HasAbstractMonad(input)) {
      monad_inputs.emplace(input);
      continue;
    }
    (*branch_input)[input] = now_input_cnt++;
  }
  for (const auto &monad_input : monad_inputs) {
    (*branch_input)[monad_input] = now_input_cnt++;
  }
}

CNodePtr ProcessSwitchNode(const KernelGraphPtr &graph, const CNodePtr &kernel_cnode,
                           std::vector<CNodePtr> *partial_inline_cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel_cnode);
  auto input_num = common::AnfAlgo::GetInputNum(kernel_cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(input_num == kSwitchInputSize,
                             "Invalid input num of switch node: " + kernel_cnode->DebugString());
  auto cond = kernel_cnode->input(kSwitchCondIndex);
  auto true_branch = kernel_cnode->input(kSwitchBranchTrueIndex);
  auto false_branch = kernel_cnode->input(kSwitchBranchFalseIndex);
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(true_branch, prim::kPrimPartialInline),
                             "Invalid true branch of switch node: " + kernel_cnode->DebugString());
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(false_branch, prim::kPrimPartialInline),
                             "Invalid false branch of switch node: " + kernel_cnode->DebugString());
  auto true_branch_cnode = true_branch->cast<CNodePtr>();
  auto false_branch_cnode = false_branch->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(true_branch_cnode);
  MS_EXCEPTION_IF_NULL(false_branch_cnode);
  std::map<AnfNodePtr, size_t> branch_input;
  CollectInputByBranchCNode(true_branch_cnode, false_branch_cnode, &branch_input);
  std::map<AnfNodePtr, AnfNodePtr> branch_tuple_getitem;
  auto cond_switch_node = GetCondSwitchNode(graph, branch_input, cond, &branch_tuple_getitem);
  MS_EXCEPTION_IF_NULL(cond_switch_node);
  auto true_branch_node = GetBranchNode(graph, true_branch_cnode, branch_tuple_getitem);
  auto false_branch_node = GetBranchNode(graph, false_branch_cnode, branch_tuple_getitem);
  auto cond_gather_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionGather->name())), true_branch_node,
                     false_branch_node});
  cond_gather_node->set_abstract(kernel_cnode->abstract());
  SelectKernelInfo(graph, cond_gather_node);
  partial_inline_cnode->emplace_back(true_branch_node);
  partial_inline_cnode->emplace_back(false_branch_node);

  // Record the branch info for condition node.
  auto false_sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(false_branch_cnode, kAttrKernelGraph);
  auto true_sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(true_branch_cnode, kAttrKernelGraph);
  MS_EXCEPTION_IF_NULL(false_sub_graph);
  MS_EXCEPTION_IF_NULL(true_sub_graph);
  std::vector<ValuePtr> branch_graph_names;
  branch_graph_names.emplace_back(std::make_shared<StringImm>(false_sub_graph->ToString()));
  branch_graph_names.emplace_back(std::make_shared<StringImm>(true_sub_graph->ToString()));
  cond_switch_node->AddAttr(kInlineSubGraphName, std::make_shared<ValueTuple>(branch_graph_names));
  reverse(branch_graph_names.begin(), branch_graph_names.end());
  cond_gather_node->AddAttr(kAttrBranchGraphName, std::make_shared<ValueTuple>(branch_graph_names));
  graph->AddConditionGatherSwitchPair(cond_gather_node, cond_switch_node);
  MS_LOG(DEBUG) << "Add new condition gather node:" << cond_gather_node->fullname_with_scope()
                << " and condition switch actor:" << cond_switch_node->fullname_with_scope()
                << " for graph:" << graph->ToString();
  graph->FrontBackendlMapUpdate(kernel_cnode, cond_switch_node);
  MS_LOG(DEBUG) << "Update backend from:" << kernel_cnode->DebugString() << " to:" << cond_switch_node->DebugString()
                << " for front node:"
                << (graph->GetFrontAnfByBackendAnf(cond_switch_node) == nullptr
                      ? "null"
                      : graph->GetFrontAnfByBackendAnf(cond_switch_node)->DebugString());
  return cond_gather_node;
}

// Flatten the input abstract, and record the index construct.
// eg. tuple(tuple(1, 2), 3) -> {1, 2, 3}  ((-1, -1), -1)
AbstractBasePtrList CollectAbstract(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return {abstract};
  }
  const auto &seq_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    return {seq_abs};
  }

  AbstractBasePtrList abs_list;
  for (const auto &sub_abs : seq_abs->elements()) {
    AbstractBasePtrList sub_list = CollectAbstract(sub_abs);
    abs_list.insert(abs_list.end(), sub_list.begin(), sub_list.end());
  }
  return abs_list;
}

size_t ConstructAbstructIndex(const abstract::AbstractBasePtr &abstract, ValuePtr *abstract_construct_index) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    *abstract_construct_index = MakeValue(-1);
    return 1;
  }
  const auto &seq_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    *abstract_construct_index = MakeValue(-1);
    return 1;
  }

  ValuePtrList construct_index_list;
  size_t index_num = 0;
  for (const auto &sub_abs : seq_abs->elements()) {
    ValuePtr sub_construct_index = nullptr;
    index_num += ConstructAbstructIndex(sub_abs, &sub_construct_index);
    construct_index_list.emplace_back(sub_construct_index);
  }
  *abstract_construct_index = std::make_shared<ValueTuple>(construct_index_list);
  return index_num;
}

// Rebuild the output construct by construct index.
CNodePtr ConstructMakeTupleRecursion(const ValuePtr &abstract_construct_index, std::deque<CNodePtr> *get_item_list,
                                     const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  MS_EXCEPTION_IF_NULL(get_item_list);
  MS_EXCEPTION_IF_NULL(graph);
  if (!abstract_construct_index->isa<ValueSequence>()) {
    if (get_item_list->empty()) {
      MS_LOG(EXCEPTION) << "Failed to get item node by value:" << abstract_construct_index->ToString();
    } else {
      auto top = get_item_list->front();
      get_item_list->pop_front();
      return top;
    }
  }

  // Build node and abstract for tuple construct.
  const auto &seq_value = abstract_construct_index->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_value);
  AnfNodePtrList node_list{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  AbstractBasePtrList abs_list;
  for (const auto &sub_value : seq_value->value()) {
    MS_EXCEPTION_IF_NULL(sub_value);
    const auto &new_node = ConstructMakeTupleRecursion(sub_value, get_item_list, graph);
    MS_EXCEPTION_IF_NULL(new_node);
    node_list.emplace_back(new_node);
    abs_list.emplace_back(new_node->abstract());
  }
  const auto &make_tuple = graph->NewCNode(node_list);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  return make_tuple;
}

AnfNodePtrList CreateTupleGetItemForTupleOutput(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract = node->abstract();
  if (abstract == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid abstract for node:" << node->DebugString();
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    return {node};
  }
  const auto &sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abstract);
  if (sequence_abstract->dynamic_len()) {
    return {node};
  }
  AnfNodePtrList outputs;
  for (size_t i = 0; i < sequence_abstract->elements().size(); ++i) {
    const auto &sub_abstract = sequence_abstract->elements()[i];
    MS_EXCEPTION_IF_NULL(sub_abstract);
    auto get_item = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), node,
                                     NewValueNode(MakeValue<int64_t>(SizeToLong(i)))});
    get_item->set_abstract(sub_abstract);
    const auto &sub_outputs = CreateTupleGetItemForTupleOutput(get_item, graph);
    outputs.insert(outputs.end(), sub_outputs.begin(), sub_outputs.end());
  }
  return outputs;
}

// Flatten the tuple input of condition gather.
CNodePtr FlattenConditionGatherNodeInput(const CNodePtr &kernel, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  if (mng == nullptr) {
    auto manager = MakeManager({graph});
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
    mng = graph->manager();
  }

  AnfNodePtrList new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionGather->name()))};
  size_t output_num = SIZE_MAX;
  // Collect inputs.
  for (size_t i = 1; i < kernel->inputs().size(); ++i) {
    const auto &input = kernel->inputs()[i];
    MS_EXCEPTION_IF_NULL(input);
    AnfNodePtrList outputs = CreateTupleGetItemForTupleOutput(input, graph);
    // All input branch should have same output num.
    if (output_num != SIZE_MAX && output_num != outputs.size()) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid output size:" << output_num << " and " << outputs.size()
                                          << " for kernel:" << kernel->fullname_with_scope();
    }
    output_num = outputs.size();
    new_inputs.insert(new_inputs.end(), outputs.begin(), outputs.end());
  }

  // Create new condition gather node.
  auto new_kernel = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_kernel);
  ValuePtr abstract_construct_index = nullptr;
  AbstractBasePtrList new_abstract_list = CollectAbstract(kernel->abstract());
  auto front_abs = kernel->abstract();
  auto gather_to_switch = graph->condition_gather_to_switch();
  const auto &switch_iter = gather_to_switch.find(kernel);
  if (switch_iter == gather_to_switch.end() || switch_iter->second == nullptr) {
    MS_LOG(WARNING) << "Failed to get condition switch node by condition gather:" << kernel->DebugString();
  } else {
    const auto &front_call = graph->GetFrontAnfByBackendAnf(switch_iter->second);
    if (front_call == nullptr || front_call->abstract() == nullptr) {
      MS_LOG(WARNING) << "Failed to get front call node by switch node:" << switch_iter->second->DebugString();
    } else {
      front_abs = front_call->abstract();
      MS_LOG(DEBUG) << "Rebuild output by front call node:" << front_call->DebugString()
                    << " abstract:" << front_abs->ToString()
                    << " by condition switch node:" << switch_iter->second->fullname_with_scope();
    }
  }
  size_t index_num = ConstructAbstructIndex(front_abs, &abstract_construct_index);
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  MS_LOG(INFO) << "Abstract construct index:" << abstract_construct_index->ToString()
               << " for rebuild the abstract of kernel:" << new_kernel->DebugString();
  if (new_abstract_list.size() != output_num || output_num != index_num) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid abstract list size:" << new_abstract_list.size()
                                        << " and output size:" << output_num << " output index size:" << index_num
                                        << " for kernel:" << kernel->DebugString()
                                        << " abstract:" << kernel->abstract()->ToString();
  }
  new_kernel->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abstract_list));
  SelectKernelInfo(graph, new_kernel);
  if (output_num == SIZE_MAX) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid output size:" << output_num
                                        << " for kernel:" << kernel->fullname_with_scope();
  }
  new_kernel->AddAttr(kAttrBranchOutputNum, MakeValue<size_t>(output_num));
  if (kernel->HasAttr(kAttrBranchGraphName)) {
    new_kernel->AddAttr(kAttrBranchGraphName, kernel->GetAttr(kAttrBranchGraphName));
  }

  // Rebuild the output construct for condition gather node.
  std::deque<CNodePtr> get_item_list;
  for (size_t i = 0; i < output_num; ++i) {
    auto get_item = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())),
                                     new_kernel, NewValueNode(MakeValue<int64_t>(i))});
    MS_EXCEPTION_IF_NULL(get_item);
    get_item_list.emplace_back(get_item);
    get_item->set_abstract(new_abstract_list[i]);
  }
  auto make_tuple = ConstructMakeTupleRecursion(abstract_construct_index, &get_item_list, graph);
  (void)mng->Replace(kernel, make_tuple);
  return new_kernel;
}

// Flatten the tuple input of condition node.
void FlattenConditionNodeInput(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_flatten_gather_input_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  const auto &nodes = TopoSort(graph->output());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto kernel = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(kernel);
    if (IsPrimitiveCNode(kernel, prim::kPrimConditionSwitch)) {
      for (size_t i = 2; i < kernel->size(); ++i) {
        const auto &input = kernel->input(i);
        MS_EXCEPTION_IF_NULL(input);
        if ((!input->isa<Parameter>()) || HasAbstractMonad(input) || common::AnfAlgo::HasAbstractRef(input)) {
          continue;
        }
        std::vector<AnfNodePtr> tensor_move_inputs = {
          NewValueNode(std::make_shared<Primitive>(prim::kPrimTensorMove->name())), input};
        auto tensor_move = graph->NewCNode(tensor_move_inputs);
        tensor_move->set_abstract(input->abstract()->Clone());
        kernel->set_input(i, tensor_move);
      }
    }
    if (!IsPrimitiveCNode(kernel, prim::kPrimConditionGather)) {
      continue;
    }
    const auto &new_kernel = FlattenConditionGatherNodeInput(kernel, graph);
    MS_EXCEPTION_IF_NULL(new_kernel);
    auto gather_to_switch = graph->condition_gather_to_switch();
    const auto &iter = gather_to_switch.find(kernel);
    if (iter == gather_to_switch.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Failed to get condition switch node for gather:" << kernel->DebugString();
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    const auto &inline_iter = graph->inline_sub_graph_kernels().find(kernel);
    if (inline_iter != graph->inline_sub_graph_kernels().end()) {
      auto subgraph_name = inline_iter->second;
      graph->AddInlineSubgraphKernel(new_kernel, subgraph_name);
      MS_LOG(INFO) << "Add new condition gather node:" << new_kernel->fullname_with_scope()
                   << " subgraph name:" << subgraph_name << " to graph:" << graph->ToString();
    }
    graph->AddConditionGatherSwitchPair(new_kernel, iter->second);
    graph->RemoveConditionGatherSwitchPair(kernel);
    MS_LOG(INFO) << "Add new condition gather node:" << new_kernel->fullname_with_scope()
                 << " to replace node:" << kernel->fullname_with_scope() << " branch name:"
                 << (kernel->HasAttr(kAttrBranchGraphName) ? new_kernel->GetAttr(kAttrBranchGraphName)->ToString()
                                                           : " null")
                 << " in graph:" << graph->ToString();
  }

#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_flatten_gather_input_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

void InlineSwitchGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  PROF_START(InlineSwitchGraph);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    InlineSwitchGraph(child_graph.lock(), memo);
  }
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_switch_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  // process ConditionSwitch/ConditionGather
  graph->SetExecOrderByDefault();
  auto kernel_cnodes = graph->execution_order();
  auto mng = graph->manager();
  std::vector<CNodePtr> partial_inline_cnode;
  for (auto &kernel_cnode : kernel_cnodes) {
    if (!IsPrimitiveCNode(kernel_cnode, prim::kPrimSwitch)) {
      continue;
    }
    auto cond_gather_node = ProcessSwitchNode(graph, kernel_cnode, &partial_inline_cnode);
    if (mng == nullptr) {
      auto manager = MakeManager({graph});
      MS_EXCEPTION_IF_NULL(manager);
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
      mng = manager;
    }
    (void)mng->Replace(kernel_cnode, cond_gather_node);
  }

  // inline switch graph
  std::vector<FuncGraphManagerPtr> subgraph_managers;
  for (auto &kernel_cnode : partial_inline_cnode) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimPartialInline)) {
      auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      auto sub_mng = inline_subgraph->manager();
      if (sub_mng == nullptr) {
        auto sub_manager = MakeManager({inline_subgraph}, false);
        MS_EXCEPTION_IF_NULL(sub_manager);
        sub_manager->AddFuncGraph(inline_subgraph);
        inline_subgraph->set_manager(sub_manager);
        subgraph_managers.emplace_back(sub_manager);
      }
      InlineSubGraph(graph, inline_subgraph, kernel_cnode, nullptr, nullptr, true);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, kernel_cnode) << "Invalid node type, node: " << kernel_cnode->fullname_with_scope();
    }
  }
  FlattenConditionNodeInput(graph);
  PROF_END(InlineSwitchGraph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_switch_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

std::string GetBranchName(const KernelGraphPtr &graph, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(graph);
  std::string current_branch = graph->ToString();
  const auto &iter = graph->inline_sub_graph_kernels().find(kernel);
  if (iter != graph->inline_sub_graph_kernels().end()) {
    current_branch = iter->second;
  }
  return current_branch;
}

std::set<std::string> GetSwitchBranchName(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return {};
  }
  const auto &kernel = node->cast<CNodePtr>();
  if (kernel == nullptr || !common::AnfAlgo::CheckPrimitiveType(kernel, prim::kPrimConditionSwitch) ||
      !kernel->HasAttr(kInlineSubGraphName)) {
    return {};
  }
  const auto &inline_sub_graph_names = kernel->GetAttr(kInlineSubGraphName);
  if (inline_sub_graph_names == nullptr || !inline_sub_graph_names->isa<ValueTuple>()) {
    return {};
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::set<std::string> branch_names;
  for_each(tuple_name->value().begin(), tuple_name->value().end(),
           [&branch_names](const auto &value) { branch_names.emplace(GetValue<std::string>(value)); });
  return branch_names;
}

// Put the kernels belonging to the same inline subgraph together in the execution order.
void FixExecutionOrderForInlineControlFlowGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->condition_gather_to_switch().empty()) {
    return;
  }
  auto execution_order = graph->execution_order();
  for (const auto &condition_node_pair : graph->condition_gather_to_switch()) {
    std::vector<CNodePtr> new_order;
    std::vector<CNodePtr> new_order_after_switch;
    MS_EXCEPTION_IF_NULL(condition_node_pair.first);
    MS_EXCEPTION_IF_NULL(condition_node_pair.second);
    std::set<std::string> switch_branchs = GetSwitchBranchName(condition_node_pair.second);
    std::string current_branch = GetBranchName(graph, condition_node_pair.second->cast<CNodePtr>());
    bool is_get_switch = false;
    for (auto iter = execution_order.begin(); iter != execution_order.end(); ++iter) {
      if (*iter == condition_node_pair.second) {
        is_get_switch = true;
        continue;
      }
      if (*iter == condition_node_pair.first) {
        if (!is_get_switch) {
          MS_LOG_WITH_NODE(EXCEPTION, condition_node_pair.first)
            << "Condition gather:" << condition_node_pair.first->fullname_with_scope()
            << " is in front of condition switch: " << condition_node_pair.second->fullname_with_scope();
        }
        new_order.emplace_back(condition_node_pair.second->cast<CNodePtr>());
        new_order.insert(new_order.end(), new_order_after_switch.begin(), new_order_after_switch.end());
        new_order.insert(new_order.end(), iter, execution_order.end());
        break;
      }
      // In subgraph, the input of cnode maybe empty or all of them are value node, it maybe in front of the switch node
      // in toposort.
      if (!is_get_switch && switch_branchs.find(GetBranchName(graph, *iter)) != switch_branchs.end()) {
        MS_LOG(INFO) << "Kernel:" << (*iter)->fullname_with_scope() << " reorder behind the condition switch node.";
        new_order_after_switch.emplace_back(*iter);
      } else if (!is_get_switch || current_branch == GetBranchName(graph, *iter)) {
        new_order.emplace_back(*iter);
      } else {
        new_order_after_switch.emplace_back(*iter);
      }
    }
    if (execution_order.size() != new_order.size()) {
      MS_LOG(EXCEPTION) << "Failed to reorder execution kernel for graph:" << graph->ToString();
    }
    execution_order = new_order;
  }
  graph->set_execution_order(execution_order);
}

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}
}  // namespace

void GeKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  device::ascend::AscendHalManager::GetInstance().InitializeAcl();
  MS_EXCEPTION_IF_NULL(device_context_);
  res_manager_ = device_context_->device_res_manager_.get();
  MS_EXCEPTION_IF_NULL(res_manager_);
  initialized_ = true;
}

void GeKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  initialized_ = false;
}

void GeKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  GEGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void GeKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  GEGraphOptimization::GetInstance().GEMindIRPass(graph);
}

void GeKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  std::set<KernelGraphPtr> memo;
  GEGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph, &memo);
  memo.clear();
  std::vector<size_t> op_selected_num(device::ascend::SelectedKernelType::NUM_KERNLE_TYPE, 0);
  bool has_aclop = false;
  SelectKernel(kernel_graph, &memo, &op_selected_num, &has_aclop);
  if (has_aclop) {
    static std::once_flag ge_init_flag_ = {};
    std::call_once(ge_init_flag_, [&]() {
      dynamic_cast<AscendDeviceContext *>(device_context_)->InitializeForAclop();
      SetAclOpPrecisionMode();
      res_manager_->SetAclDeterministic();
    });
  }
  PrintOpSelectedNum(op_selected_num);
  memo.clear();
  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterKernelSelect(kernel_graph, &memo);
  memo.clear();
  InlineCallGraph(kernel_graph);
  memo.clear();
  InlineSwitchGraph(kernel_graph, &memo);
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void GeKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  if (nodes.empty()) {
    return;
  }
  auto func_graph = nodes[0]->func_graph();
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  uint64_t start_time = profiler::GetClockSyscnt();
  SetKernelInfoBeforeCreateKernel(nodes);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }

  if (!kernel_graph->is_from_cache()) {
    GEGraphOptimization::GetInstance().OptimizeACLGraphAfterCreateKernel(kernel_graph);
    OptimizeExecutionOrder(NOT_NULL(func_graph));
  } else {
    MS_LOG(INFO) << "Skip optimize after create kernel for:" << kernel_graph->ToString();
  }
  (void)profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", start_time, profiler::GetClockSyscnt(),
                                  1);
  MS_LOG(DEBUG) << "Status record: end create kernel.";
}

kernel::KernelModPtr GeKernelExecutor::CreateKernelMod(const std::string &op_name) const {
  // Note: Only support generate aclnn kernel mod current.
  auto kernel_ptr = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create(op_name);
  if (kernel_ptr == nullptr) {
    MS_LOG(WARNING) << "aclnn can't find Kernel[" << op_name << "]";
    return nullptr;
  }
  device::ascend::AclnnInit();
  return kernel_ptr;
}

void GeKernelExecutor::DoStreamAssign(
  const KernelGraphPtr &kernel_graph,
  const std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> &mock_exec_order) const {
  MS_LOG(DEBUG) << "Status record: start stream assign.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "InferBoost mode, skip assign stream.";
    return;
  }
  MS_LOG(INFO) << "Status record: start stream assign, " << kernel_graph->ToString();
  // stream assign
  if (common::IsDisableRuntimeConfig(common::kRuntimeMultiStream)) {
    MS_LOG(INFO) << "Force single stream.";
  } else {
    AclStreamAssign::GetInstance().AssignStream(NOT_NULL(kernel_graph), mock_exec_order, res_manager_);
  }
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_stream_assign_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  MS_LOG(INFO) << "Status record: end stream assign, " << kernel_graph->ToString();
}

void GeKernelExecutor::DoSomas(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // somas
  MS_LOG(DEBUG) << "Status record: start do somas.";
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    auto somas = std::make_shared<AclSomas>();
    PROF_START(somas);
    bool ret = somas->Assign(kernel_graph);
    PROF_END(somas);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << kernel_graph->graph_id()
                   << " somas size: " << kernel_graph->somas_whole_block_size();
    } else if (somas->IsSupportSomas(*kernel_graph)) {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << kernel_graph->graph_id();
    }
  }
  MS_LOG(DEBUG) << "Status record: end do somas.";
}

void GeKernelExecutor::OptimizeExecutionOrder(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  PROF_START(OptimizeExecutionOrder);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start optimize execution order. graph id: " << kernel_graph->graph_id();
  kernel_graph->SetExecOrderByDefault();
  auto execution_order = kernel_graph->execution_order();
  kernel_graph->EnableRuntimeCache();
  common::AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  kernel_graph->DisableRuntimeCache();
  kernel_graph->set_execution_order(execution_order);
  MS_LOG(DEBUG) << "Status record: end optimize execution order. graph id: " << kernel_graph->graph_id();
  FixExecutionOrderForInlineControlFlowGraph(kernel_graph);
  PROF_END(OptimizeExecutionOrder);
}

namespace {
void CreateEventKernelMod(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsOneOfPrimitiveCNode(node, {prim::kPrimStreamSend, prim::kPrimStreamRecv})) {
      continue;
    }
    device::ascend::GenerateKernelBuildInfo(node, RT_KERNEL);
    auto kernel_mod_ptr = kernel::RtOpBuild(node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
  }
}
}  // namespace

void ResetNodeIds(const KernelGraphPtr &kernel_graph) {
  if (!kernel_graph->memory_managed_by_ge()) {
    MS_LOG(INFO) << "Start reset node id";
    mindspore::HashMap<std::string, int> node_ids;
    const auto &all_nodes = mindspore::TopoSort(kernel_graph->get_return(), SuccDeeperSimple);
    for (const auto &node : all_nodes) {
      if (node != nullptr && node->isa<CNode>()) {
        const auto &cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        const auto &fullname = cnode->fullname_with_scope();
        auto op_index = fullname.rfind("-op");
        if (op_index != string::npos) {
          auto scope_prefix = fullname.substr(0, op_index);
          if (node_ids.find(scope_prefix) == node_ids.end()) {
            node_ids[scope_prefix] = 0;
          } else {
            node_ids[scope_prefix]++;
          }
          cnode->set_fullname_with_scope(scope_prefix + "-op" + std::to_string(node_ids[scope_prefix]));
        }
      }
    }
    MS_LOG(INFO) << "End reset node id";
  }
}

void GeKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &nodes = kernel_graph->execution_order();
  const auto &all_nodes = TopoSort(graph->get_return());
  MS_VLOG(VL_FLOW) << "The size of execution order: " << nodes.size();
  MS_VLOG(VL_FLOW) << "The size of all node: " << all_nodes.size();
  if (common::IsEnableRuntimeConfig(common::kRuntimeCompileStat)) {
    std::cout << "The size of execution order: " << nodes.size() << std::endl;
    std::cout << "The size of all node: " << all_nodes.size() << std::endl;
  }

  // nop op -> memcpy
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = common::AnfAlgo::IsNopNode(node);
    bool is_transpose_nop =
      (op_name == prim::kPrimTransposeD->name()) && common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    if (is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }
  ResetNodeIds({kernel_graph});
  std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> mock_exec_order;
  if (common::GetEnv("MS_ENABLE_GPTO") == "1") {
    MS_LOG(INFO) << "Current Exec Order Algo in MS Context is GPTO";
    mindspore::gpto::GPTO(res_manager_, kernel_graph, &mock_exec_order);
  }
  DoStreamAssign(kernel_graph, mock_exec_order);
  CreateEventKernelMod(kernel_graph);
  kernel_graph->PrintGraphExecuteOrder();
  DoSomas(NOT_NULL(graph));
  (void)profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", start_time,
                                  profiler::GetClockSyscnt(), 1);
}

void GeKernelExecutor::CreateEventForCache(const KernelGraphPtr &kernel_graph) const {
  AclStreamAssign::GetInstance().CreateEvent(NOT_NULL(kernel_graph));
}

bool GeKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs, void *stream) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(DEBUG) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                  << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  if (inputs[0]->size() == 0) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), outputs[0]->size(),
                                    inputs[0]->device_ptr(), inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size()
                  << " count:" << inputs[0]->size();
    return false;
  }
  return true;
}

void GeKernelExecutor::DoAsyncCkpt(const CNodePtr &kernel) const {
  static bool disable_ckpt_d2h_async = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC") != "1";
  if (MS_LIKELY(disable_ckpt_d2h_async)) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto need_async_ckpt = ms_context->get_param<bool>(MS_CTX_NEED_CKPT);
  if (execution_mode == kPynativeMode || !need_async_ckpt) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(kernel->func_graph());
  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto save_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
  auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", save_steps: " << save_steps
                << ", last_triggered_step:" << last_triggered_step;
  if (cur_step >= (last_triggered_step + save_steps) && kg != nullptr) {
    AscendDeviceResManager *ascend_res_manager = dynamic_cast<AscendDeviceResManager *>(res_manager_);
    if (SkipOrResetCopyAction()) {
      MS_LOG(INFO) << "Enable async d2h copy";
      SavePrevStepWeight(kg->GetRootWeights(), ascend_res_manager->GetCopyDataStream());
    }
    if (common::AnfAlgo::HasNodeAttr(kFromRefGraph, kernel) &&
        common::AnfAlgo::GetNodeAttr<bool>(kernel, kFromRefGraph) && SkipOrResetSyncAction()) {
      MS_LOG(INFO) << "Ref op sync once action";
      AscendStreamMng::GetInstance().SyncStream(ascend_res_manager->GetCopyDataStream());
    }
  }
}

bool GeKernelExecutor::LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                    void *stream) const {
  // launch kernel
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  DoAsyncCkpt(kernel);
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs, stream)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_EXCEPTION_IF_NULL(stream);
    if (silentcheck::ascend::SilentChecker::IsNpuAsdEnable() &&
        !silentcheck::ascend::SilentChecker::GetInstance().IsCommOpInputNotSupport() &&
        kernel->HasPrimalAttr(silentcheck::kAttrSilentCheckOpType)) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch silent check for " << kernel->fullname_with_scope();
      silentcheck::ascend::SilentChecker::GetInstance().ExecuteCheck(kernel_mod, inputs[0], stream);
    }

    if (MS_UNLIKELY(UCEException::IsEnableUCE())) {
      bool is_opt_start_kernel = OptimizerEventInfo::GetInstance().IsOptimizerStartKernelMod(kernel_mod, kernel);
      bool is_opt_end_kernel = OptimizerEventInfo::GetInstance().IsOptimizerEndKernelMod(kernel_mod, kernel);
      if (is_opt_start_kernel || is_opt_end_kernel) {
        // insert event for optimizer start and end
        OptimizerEventInfo::GetInstance().RecordEvent(is_opt_start_kernel, stream);
      }
      if (is_opt_end_kernel) {
        // skip execute TensorReport op at the end of optimzer, it is just used as a tag
        return true;
      }
    }

    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      SetUceError();
      SetArfError();
      return false;
    }
  }
  // for PyNative and KBK Sync Run mode
  if (MS_UNLIKELY(mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking())) {
    if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel)
        << "Sync run failed, detail: " << CALL_ASCEND_API(aclGetRecentErrMsg) << trace::DumpSourceLines(kernel);
    }
  }
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false, inputs);
  return true;
}

bool GeKernelExecutor::LaunchKernelHP(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                      void *stream) const {
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs, stream)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    if (silentcheck::ascend::SilentChecker::IsNpuAsdEnable() &&
        !silentcheck::ascend::SilentChecker::GetInstance().IsCommOpInputNotSupport() &&
        kernel->HasPrimalAttr(silentcheck::kAttrSilentCheckOpType)) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch silent check for " << kernel->fullname_with_scope();
      silentcheck::ascend::SilentChecker::GetInstance().ExecuteCheck(kernel_mod, inputs[0], stream);
    }

    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      SetUceError();
      SetArfError();
      return false;
    }
  }
  return true;
}

void GeKernelExecutor::SetUceError() const {
  if (UCEException::IsEnableUCE() && aclrt_get_last_error != nullptr) {
    auto error_code = aclrt_get_last_error(thread_level);
    auto error_type = GetErrorType(error_code);
    UCEException::GetInstance().ProcessUceError(
      mindspore::FuncInfo{FILE_NAME, __LINE__, __FUNCTION__, "Launch kernel failed"}, error_code,
      acl_get_recent_err_msg, error_type);
    if (!UCEException::GetInstance().get_has_throw_error()) {
      auto arf_env = common::GetEnv("MS_ENABLE_TFT");
      constexpr std::string_view arf = "ARF:1";
      if (arf_env.find(arf) == std::string::npos) {
        res_manager_->ResetStreamAndCtx();
      }
    }
  }
}

void GeKernelExecutor::SetArfError() const {
  if (UCEException::GetInstance().enable_arf()) {
    auto rts_code = aclrt_get_last_error(thread_level);
    MS_LOG(ERROR) << "Launch kernel failed, get last error is " << rts_code;
    if (rts_code == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
      UCEException::GetInstance().set_force_stop_flag(true);
    }
  }
}

void AclrtLaunchCallback(void *user_data) {
  CallbackFunc *callback_func = reinterpret_cast<CallbackFunc *>(user_data);
  (*callback_func)();
  delete callback_func;
}

bool GeKernelExecutor::LaunchCallback(CallbackFunc callback_func, size_t stream_id, bool is_block) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream_id = kDefaultStreamIndex;
    stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  }
  MS_EXCEPTION_IF_NULL(stream);
  auto block_type =
    is_block ? aclrtCallbackBlockType::ACL_CALLBACK_BLOCK : aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK;
  auto callback_func_ptr = new CallbackFunc(callback_func);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callback_func_ptr, block_type, stream);
  MS_LOG(DEBUG) << "Launch callback for stream_id : " << stream_id << ", ret : " << ret << ".";
  if (ret) {
    delete callback_func_ptr;
    MS_LOG(ERROR) << "Launch callback for stream_id : " << stream_id << " failed, ret : " << ret << ".";
    if (res_manager_->SyncStream(stream_id)) {
      callback_func();
      return true;
    }
    auto arf_env = common::GetEnv("MS_ENABLE_TFT");
    constexpr std::string_view arf = "ARF:1";
    if (arf_env.find(arf) == std::string::npos) {
      res_manager_->ResetStreamAndCtx();
    }
    return false;
  }
  return true;
}

void CustomizeCopyAscendInner(device::DeviceContext *device_context, const device::DeviceAddressPtr &input_addr,
                              const device::DeviceAddressPtr &output_addr, const size_t &stream_id) {
  // The input_addr_list address is malloc before
  // Malloc for output tensors
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", device::tracker::MemType::kPyNativeOutput,
                                                 output_addr->GetSize(), output_addr.get());
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsInput, "PyNative", input_addr->device_name(),
                                                 input_addr->GetPtr(), input_addr->type_id(),
                                                 input_addr->GetShapeVector(), input_addr->GetTensorStorageInfo());
  if (output_addr->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(output_addr.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }
  const auto &input_storage_info = input_addr->address_common()->tensor_storage_info_;
  const auto &output_storage_info = output_addr->address_common()->tensor_storage_info_;
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", output_storage_info:" << (output_storage_info == nullptr ? "" : output_storage_info->ToString())
                << ", input address size:" << input_addr->GetSize()
                << ", output address size:" << output_addr->GetSize();

  // Inplace output need be front
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, stream_id, output_addr, input_addr);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PyNative", output_addr->device_name(),
                                                 output_addr->GetPtr(), output_addr->type_id(),
                                                 output_addr->GetShapeVector(), output_addr->GetTensorStorageInfo());
  MS_LOG(DEBUG) << "Launch end";
}

// Unconventional pyboost writing. Please do not refer to this to implement other operators!
void CustomizeCopyAscend(device::DeviceContext *device_context, const device::DeviceAddressPtr &input_addr,
                         const device::DeviceAddressPtr &output_addr, const size_t &stream_id) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(output_addr);

  if (runtime::Pipeline::Get().backend_stage()->CanPush()) {
    MS_LOG(DEBUG) << "Dispatch inplacecopy to backend queue";
    kernel::pyboost::PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([device_context, input_addr, output_addr, stream_id]() {
        CustomizeCopyAscendInner(device_context, input_addr, output_addr, stream_id);
      }));
    return;
  }

  runtime::Pipeline::Get().WaitForward();
  CustomizeCopyAscendInner(device_context, input_addr, output_addr, stream_id);
  MS_LOG(DEBUG) << "Launch end";
}

bool GeKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                         const device::DeviceAddressPtrList &input_addr_list,
                                         const device::DeviceAddressPtrList &output_addr_list,
                                         const size_t &stream_id) const {
  MS_LOG(DEBUG) << "task_type:" << task_type;
  if (runtime::KernelTaskType::kCOPY_TASK == task_type) {
    constexpr size_t kCopyTaskInputsNum = 2;
    // Copy task is a in-place op, the output is the first input.
    // To reuse the aclnnInplaceCopy, the first input of Copy is used as the operator output,
    // and the second input is used as the operator input.
    if (input_addr_list.size() != kCopyTaskInputsNum) {
      MS_LOG(EXCEPTION) << "input_addr_list.size() is invalid, input_addr_list.size():" << input_addr_list.size();
    }
    CustomizeCopyAscend(device_context_, input_addr_list[1], input_addr_list[0], stream_id);
  } else {
    // For contiguous task, there must be at least one input and one output.
    if (input_addr_list.empty() || output_addr_list.empty()) {
      MS_LOG(EXCEPTION) << "input_addr_list.size() or output_addr_list.size() is invalid, input_addr_list.size():"
                        << input_addr_list.size() << ", output_addr_list.size():" << output_addr_list.size();
    }
    if (std::dynamic_pointer_cast<device::cpu::CPUDeviceAddress>(input_addr_list[0]) != nullptr &&
        std::dynamic_pointer_cast<device::cpu::CPUDeviceAddress>(output_addr_list[0]) != nullptr) {
      // for unrefmode, the output is on host, just copy the device ptr
      output_addr_list[0]->set_ptr(input_addr_list[0]->GetMutablePtr());
    } else {
      CustomizeCopyAscend(device_context_, input_addr_list[0], output_addr_list[0], stream_id);
    }
  }

  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  // for PyNative and KBK Sync Run mode
  bool ret = true;
  if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
    ret = AscendStreamMng::GetInstance().SyncStream(stream);
  }
  return ret;
}

// Task for graph mode, which receive pointers as input, it is used for convert input contiguous by aclnnInplaceCopy.
bool GeKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                         const std::vector<device::DeviceAddress *> &input_addr_list,
                                         const std::vector<device::DeviceAddress *> &output_addr_list,
                                         const size_t &stream_id) const {
  MS_LOG(DEBUG) << "task_type:" << task_type;
  MS_LOG(DEBUG) << "Graph call contiguous start";
  auto input_addr = input_addr_list[0];
  auto output_addr = output_addr_list[0];
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(output_addr);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kPyNativeOutput,
                                                 output_addr->GetSize(), output_addr);
  if (output_addr->GetPtr() == nullptr) {
    if (!device_context_->device_res_manager_->AllocateMemory(output_addr)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }
  const auto &input_storage_info = input_addr->address_common()->tensor_storage_info_;
  const auto &output_storage_info = output_addr->address_common()->tensor_storage_info_;
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", output_storage_info:" << (output_storage_info == nullptr ? "" : output_storage_info->ToString())
                << ", input address size:" << input_addr->GetSize()
                << ", output address size:" << output_addr->GetSize();
  auto stream_ptr = device_context_->device_res_manager_->GetStream(stream_id);
  auto res = GEN_EXECUTOR_FOR_RESIZE(std::string("aclnnInplaceCopy"), output_addr, input_addr);
  auto workspace_size = std::get<0>(res);
  auto executor = std::get<1>(res);
  std::function<void()> release_func{nullptr};
  if (workspace_size == 0) {
    RUN_OP_API_ASYNC(std::string("aclnnInplaceCopy"), nullptr, 0, executor, stream_ptr, release_func);
  } else {
    auto workspace_addr = device_context_->device_res_manager_->AllocateMemory(workspace_size);
    RUN_OP_API_ASYNC(std::string("aclnnInplaceCopy"), workspace_addr, workspace_size, executor, stream_ptr,
                     release_func);
    device_context_->device_res_manager_->FreeMemory(workspace_addr);
  }
  MS_LOG(DEBUG) << "Graph call contiguous end";
  return true;
}
}  // namespace mindspore::device::ascend
