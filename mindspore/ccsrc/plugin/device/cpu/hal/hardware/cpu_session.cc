/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/hardware/cpu_session.h"
#include <algorithm>
#include <sstream>
#include <exception>
#include "ir/anf.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "common/ms_factory.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/optimizer/print_value_type.h"
#ifdef ENABLE_AKG
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.h"
#endif
#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "plugin/device/cpu/optimizer/insert_cast_cpu.h"
#include "plugin/device/cpu/optimizer/insert_cast_to_pyexecute.h"
#include "plugin/device/cpu/optimizer/insert_format_transform_op.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "backend/common/pass/replace_node_by_proxy.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "include/common/debug/anf_ir_dump.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "include/common/debug/dump_proto.h"
#include "kernel/graph_kernel_info.h"
#include "kernel/framework_utils.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_context.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
#endif

namespace mindspore {
namespace device::cpu {
void SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel, const std::vector<kernel::KernelAttr> &apply_kernel_attrs);
}
namespace session {
void CPUSession::Init(uint32_t device_id) {
  // Dump json config file if dump is enabled
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyMSCfgJsonToDir(rank_id_);
  InitExecutor(kCPUDevice, device_id);
}

ParameterPtr CPUSession::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG_WITH_NODE(EXCEPTION, anf) << "anf[" << anf->DebugString() << "] is not a parameter";
  }
  auto valid_inputs = graph->MutableValidInputs();
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  MS_EXCEPTION_IF_NULL(graph_inputs);
  TraceManager::DebugTrace(MakeTraceInfo<TraceCopy>(anf->debug_info()));
  ParameterPtr new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  TraceManager::EndTrace();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(true);
  return new_parameter;
}

// Remove after PS feature finish adapting push/pull in auto_monad.
void CPUSession::Reorder(std::vector<CNodePtr> *node_list) const {
  common::AnfAlgo::ReorderPosteriorExecList(NOT_NULL(node_list));
}

void CPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
#if defined(__linux__) && defined(WITH_BACKEND)
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ps::PSContext::instance()->is_ps_mode()) {
    if (ps::PSContext::instance()->is_worker()) {
      std::string pass_name = "replace_node_by_proxy";
      pass_name.append(std::to_string(graph_sum_));
      pm->AddPass(std::make_shared<opt::ReplaceNodeByProxy>(pass_name));
    }
  }
#endif
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOpCPU>("insert_format_transform_op_cpu"));
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast"));
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
  pm->AddPass(std::make_shared<opt::PrintValueType>("print_value_type"));
  pm->AddPass(std::make_shared<opt::InsertCastToPyExecute>("insert_cast_for_pyexecute"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void CPUSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  graphkernel::GraphKernelOptimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void CPUSession::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                     VectorRef *outputs,
                                     std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                                     KernelMapTensor *) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  runtime_.CreateOutputTensors(kernel_graph.get(), input_tensors, outputs, tensor_to_node);
}

void CPUSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                               const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &input_nodes = kernel_graph->input_nodes();
  if (input_nodes.size() != inputs_const.size()) {
    MS_LOG(EXCEPTION) << "Input size " << inputs_const.size() << " is not equal to input node size "
                      << input_nodes.size();
  }
  for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
    auto &input_node = input_nodes[input_idx];
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<Parameter>() || HasAbstractMonad(input_node)) {
      continue;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    auto tensor = inputs_const[input_idx];
    MS_EXCEPTION_IF_NULL(address);
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_address = tensor->device_address();
    if (tensor_address == nullptr || tensor_address == address) {
      continue;
    }
    auto input_param = input_node->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(input_param) && !tensor->IsUpdatedByDevice()) {
      continue;
    }
    if (std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address)->GetDeviceType() != device::DeviceType::kCPU) {
      tensor->data_sync(false);
    }
  }
}

void CPUSession::PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs, VectorRef *const outputs) {
  MS_LOG(INFO) << "Bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);
}

void CPUSession::PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                  const std::vector<tensor::TensorPtr> &, VectorRef *const) {
  Summary(kernel_graph.get());
}

void CPUSession::ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) {
  bool ret = runtime_.Run(*kernel_graph, false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run graph failed";
  }
}

void CPUSession::SetOutputFlags(const VectorRef &base_ref) {
  for (size_t i = 0; i < base_ref.size(); ++i) {
    if (utils::isa<VectorRef>(base_ref[i])) {
      auto ref_iter = utils::cast<VectorRef>(base_ref[i]);
      SetOutputFlags(ref_iter);
    } else if (utils::isa<tensor::TensorPtr>(base_ref[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref[i]);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      tensor_ptr->data_sync(false);
    }
  }
}

void CPUSession::UpdateDynamicOutputShape(const std::map<tensor::TensorPtr, KernelWithIndex> &tensor_to_node) const {
  for (const auto &tensor_node : tensor_to_node) {
    if (common::AnfAlgo::IsDynamicShape(tensor_node.second.first)) {
      const auto &kernel = tensor_node.second.first;
      const auto &output_index = tensor_node.second.second;
      const auto &shape = common::AnfAlgo::GetOutputInferShape(kernel, output_index);
      MS_EXCEPTION_IF_NULL(tensor_node.first);
      (void)tensor_node.first->set_shape(shape);
    }
  }
}

void CPUSession::SetKernelInfo(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kCPUDevice);
  MS_EXCEPTION_IF_NULL(kernel_info_setter);
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_info_setter->SetKernelInfo(kernel_node, KernelType::UNKNOWN_KERNEL_TYPE);
  }
}

namespace {
void KernelNotSupportException(const AnfNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  std::stringstream operator_info;
  operator_info << "Operator[" << kernel_name << "] ";
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_node->kernel_info());
  if (kernel_info == nullptr) {
    operator_info << "is not support.";
    MS_LOG(EXCEPTION) << operator_info.str();
  }
  auto kernel_build_Info = kernel_info->select_kernel_build_info();
  if (kernel_build_Info == nullptr) {
    operator_info << "is not support.";
    MS_LOG(EXCEPTION) << operator_info.str();
  }
  size_t input_num = kernel_build_Info->GetInputNum();
  if (input_num > 0) {
    operator_info << " input(";
    for (size_t i = 0; i < input_num; ++i) {
      operator_info << TypeIdLabel(kernel_build_Info->GetInputDeviceType(i));
      if (i != input_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  size_t output_num = kernel_build_Info->GetOutputNum();
  if (output_num > 0) {
    operator_info << "output(";
    for (size_t i = 0; i < output_num; ++i) {
      operator_info << TypeIdLabel(kernel_build_Info->GetOutputDeviceType(i));
      if (i != kernel_build_Info->GetOutputNum() - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  operator_info << "is not support.";
  MS_LOG_WITH_NODE(EXCEPTION, kernel_node) << operator_info.str() << trace::DumpSourceLines(kernel_node);
}
}  // namespace

void CPUSession::BuildKernel(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  std::vector<AnfNodePtr> akg_nodes;
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "Cpu building operator[" << kernel_name << "].";
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel_node) == KernelType::AKG_KERNEL) {
      if (!bin_map->initialized()) {
        bin_map->Initialize();
      }
      akg_nodes.push_back(kernel_node);
      continue;
    }
    std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel_mod =
      kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(kernel_name);
    if (cpu_kernel_mod == nullptr) {
      KernelNotSupportException(kernel_node);
    }

    auto kernel_attrs = cpu_kernel_mod->GetOpSupport();
    device::cpu::SetCpuRefMapToKernelInfo(kernel_node, kernel_attrs);
    auto inputs = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel_node);
    auto outputs = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel_node);
    auto ret = cpu_kernel_mod->Init(inputs, outputs);
    if (!ret) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel_node) << trace::DumpSourceLines(kernel_node);
    }
    if (cpu_kernel_mod->Resize(inputs, outputs) == static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel_node)
        << "CPU kernel op [" << kernel_node->fullname_with_scope() << "] Resize failed.";
    }
    AnfAlgo::SetKernelMod(cpu_kernel_mod, kernel_node.get());
    MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
  }
#ifdef ENABLE_AKG
  kernel::AkgCpuKernelBuilder akg_cpu_kernel_builder;
  (void)akg_cpu_kernel_builder.SingleOpParallelBuild(akg_nodes);
#endif
}
}  // namespace session
}  // namespace mindspore
