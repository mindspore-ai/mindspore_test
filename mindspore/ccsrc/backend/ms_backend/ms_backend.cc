/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/ms_backend.h"

#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "mindspore/ops/op_def/structure_op_name.h"
#include "include/common/utils/parallel_context.h"
#include "backend/graph_compiler/transform.h"
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "include/backend/optimizer/helper.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/pynative/grad/jit/jit_call_graph.h"
#include "ir/anf.h"
#include "pybind_api/ir/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/callbacks.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/graph_adapter.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "pybind_api/gil_scoped_long_running.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#include "include/backend/debug/data_dump/dump_json_parser.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#endif

#include "runtime/device/device_address_utils.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/pipeline/task/run_graph_task.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
namespace {
ValuePtr GetInputofBpropCut(const std::shared_ptr<GraphCompiler> &graph_compiler, const CNodePtr &parent_node,
                            const AnfNodePtr &input_node,
                            const std::map<KernelWithIndex, tensor::BaseTensorPtr> &op_output,
                            const std::map<AnfNodePtr, size_t> &parameter_index,
                            const std::vector<TensorPtr> &graph_inputs, InputInfo *input_info, size_t input_index) {
  if (!IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
    auto real_input = common::AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(real_input);
    ValuePtr value = nullptr;
    if (!real_input->isa<ValueNode>()) {
      if (real_input->abstract() != nullptr && real_input->abstract()->isa<abstract::AbstractSparseTensor>()) {
        value = TensorListToSparseTensor(real_input->abstract(), graph_inputs);
      } else {
        value = graph_compiler->GetSingleOpInputTensorByIndex(parent_node, op_output, parameter_index, graph_inputs,
                                                              input_info, input_index);
      }
      MS_EXCEPTION_IF_NULL(value);
    } else {
      const auto &value_node = real_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
    }
    return value;
  }
  auto cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::vector<ValuePtr> args_tuple;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto input = cnode->inputs()[i];
    auto value =
      GetInputofBpropCut(graph_compiler, cnode, input, op_output, parameter_index, graph_inputs, input_info, i - 1);
    MS_EXCEPTION_IF_NULL(value);
    (void)args_tuple.emplace_back(value);
  }
  auto arg = std::make_shared<ValueTuple>(args_tuple);
  return arg;
}

ValuePtr GetFrontArgByParameter(const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                                const AnfNodePtr &front_node) {
  const auto &iter = std::find(origin_paramters.begin(), origin_paramters.end(), front_node);
  const size_t index = static_cast<size_t>(iter - origin_paramters.begin());
  // If the parameter is not found in the parameters of the root graph, it means that it is the input of the subgraph,
  // and there is no need to input a tensor.
  if (index >= front_args.size()) {
    MS_LOG(EXCEPTION) << "Position out of front args range, position value is " << index << " and args size is "
                      << front_args.size() << ".";
  }
  auto value = utils::cast<ValuePtr>(front_args[index]);
  MS_EXCEPTION_IF_NULL(value);
  return value;
}

void GetControlOpInput(const std::shared_ptr<GraphCompiler> &graph_compiler,
                       const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                       const CNodePtr &front_cnode, const CNodePtr &backend_cnode,
                       const std::map<KernelWithIndex, tensor::BaseTensorPtr> &op_output_map,
                       const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs, InputInfo *input_info, VectorRef *args) {
  MS_EXCEPTION_IF_NULL(front_cnode);
  MS_EXCEPTION_IF_NULL(backend_cnode);
  MS_EXCEPTION_IF_NULL(graph_compiler);
  MS_EXCEPTION_IF_NULL(args);
  auto front_size = front_cnode->size();
  auto back_size = backend_cnode->size();
  if (front_size != back_size) {
    MS_LOG(EXCEPTION) << "Bpropcut op front cnode size: " << front_size << ", back cnode size:" << back_size
                      << ", bpropcut op should not flatten";
  }
  for (size_t index = 1; index < back_size; ++index) {
    auto input_node = backend_cnode->input(index);
    MS_EXCEPTION_IF_NULL(input_node);
    ValuePtr value = nullptr;
    if (input_node->isa<Parameter>() && input_node->abstract() != nullptr &&
        input_node->abstract()->isa<abstract::AbstractSequence>()) {
      auto front_input_node = front_cnode->input(index);
      value = GetFrontArgByParameter(origin_paramters, front_args, front_input_node);
    } else {
      value = GetInputofBpropCut(graph_compiler, backend_cnode, input_node, op_output_map, parameter_index,
                                 graph_inputs, input_info, index - 1);
    }
    MS_EXCEPTION_IF_NULL(value);
    (void)args->emplace_back(value);
  }
}

void RunControlOperator(const std::shared_ptr<GraphCompiler> &graph_compiler,
                        const std::vector<AnfNodePtr> &origin_paramters, const VectorRef &front_args,
                        const KernelGraphPtr &graph, const CNodePtr &kernel,
                        const std::map<KernelWithIndex, tensor::BaseTensorPtr> &op_output_map,
                        const std::map<AnfNodePtr, size_t> &parameter_index,
                        const std::vector<tensor::TensorPtr> &graph_inputs, InputInfo *input_info,
                        VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_outputs);
  AnfNodePtr front_node = graph->GetFrontAnfByBackendAnf(kernel);
  if (front_node == nullptr && graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    front_node = kernel;
  }
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The front node of bprop_cut is not CNode";
  }
  CNodePtr cnode = front_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const std::vector<AnfNodePtr> &node_inputs = cnode->inputs();
  if (node_inputs.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The inputs of node[" << cnode->fullname_with_scope() << "] is empty";
  }

  const AnfNodePtr &fn = node_inputs.at(0);
  if (!IsValueNode<Primitive>(fn)) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "The input[0] of kernel[" << kernel->fullname_with_scope()
                                        << "] is not a ValueNode of Primitive";
  }

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(fn);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == kBpropCutOpName) {
    VectorRef args;
    GetControlOpInput(graph_compiler, origin_paramters, front_args, cnode, kernel, op_output_map, parameter_index,
                      graph_inputs, input_info, &args);
    py::gil_scoped_acquire acquire;
    BaseRef out = python_adapter::PyAdapterCallback::RunPrimitivePyHookFunction(prim, args);
    // Convert pyobject output to tensor.
    if (utils::isa<PyObjectRef>(out)) {
      PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
      auto out_py_tuple = py_ref.object_;
      std::vector<ValuePtr> output_tensors;
      ConvertPyObjectToCTensor(out_py_tuple, &output_tensors);
      // If bprop change grad, kernel abstract need update for its users
      std::vector<abstract::AbstractBasePtr> output_tensor_abs;
      for (auto &tensor : output_tensors) {
        (void)output_tensor_abs.emplace_back(tensor->ToAbstract()->Broaden());
        (void)op_outputs->elements_.emplace_back(std::move(tensor));
      }
      kernel->set_abstract(std::make_shared<abstract::AbstractTuple>(output_tensor_abs));
    }
  }
}
}  // namespace

void CreateKernelTensor(const std::vector<std::vector<tensor::TensorPtr>> &input_tensors,
                        std::vector<DeviceContext *> device_contexts) {
  if (input_tensors.size() < device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Invalid input_tensors size " << input_tensors.size() << " device_contexts size "
                      << device_contexts.size();
  }
  for (size_t i = 0; i < device_contexts.size(); ++i) {
    const auto &tensors = input_tensors[i];
    const auto &device_context = device_contexts[i];
    MS_EXCEPTION_IF_NULL(device_context);
    for (const auto &tensor : tensors) {
      if (tensor != nullptr && tensor->device_address() != nullptr) {
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
        MS_EXCEPTION_IF_NULL(device_address);
        if (device_address->kernel_tensor() == nullptr) {
          runtime::DeviceAddressUtils::CreateKernelTensor(device_address, tensor.get());
        }
      }
    }
  }
}

void CreateKernelTensor(const BaseRef &arg) {
  if (utils::isa<tensor::BaseTensor>(arg)) {
    auto tensor = utils::cast<tensor::BaseTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (device_address != nullptr) {
      runtime::DeviceAddressUtils::CreateKernelTensor(device_address, tensor.get());
    }
  } else if (utils::isa<ValueSequencePtr>(arg)) {
    auto value_sequence = utils::cast<ValueSequencePtr>(arg);
    MS_EXCEPTION_IF_NULL(value_sequence);
    const auto &sequence_value = value_sequence->value();
    for (const auto &value : sequence_value) {
      CreateKernelTensor(value);
    }
  } else if (utils::isa<stub::TensorNode>(arg)) {
    auto tensor_stub = utils::cast<std::shared_ptr<stub::TensorNode>>(arg);
    MS_EXCEPTION_IF_NULL(tensor_stub);
    auto value = tensor_stub->WaitValue();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (device_address != nullptr) {
      runtime::DeviceAddressUtils::CreateKernelTensor(device_address, tensor.get());
    }
  } else {
    MS_LOG(DEBUG) << "Only tensor need create KernelTensor";
  }
}

void CreateKernelTensor(const VectorRef &args) {
  for (const auto &arg : args) {
    CreateKernelTensor(arg);
  }
}

MSBackend::~MSBackend() {
  if (enable_graph_pipeline_) {
    GilReleaseWithCheck gil_release;
    runtime::Pipeline::Get().frontend_stage()->Wait();
  }
}

runtime::ActorSet *MSBackend::RealCompileGraphBeforeRunActor(BackendGraphId graph_id,
                                                             const GraphCompilerInfo &graph_compiler_info,
                                                             const VectorRef &args, bool no_multi_graph) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  ContiguousArgs(args, graph_compiler_info);
  WaitTaskFinish();
  auto graphs = graph_compiler_info.graphs_;
  auto device_contexts = graph_compiler_info.device_contexts_;
  CreateKernelTensor(args);

  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_flag(kFlagPyNativeRunInGraph, true);
    graph->set_flag(kFlagIsPynativeBpropGraph, root_graph_->has_flag(kFlagIsPynativeBpropGraph));
    if (graph->is_any_type_input()) {
      continue;
    }
    auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (enable_graph_pipeline_) {
      for (const auto &tensors : input_tensors) {
        for (const auto &tensor : tensors) {
          if (tensor) {
            tensor->set_need_pipeline_sync(true);
          }
        }
      }
    }

    if (no_multi_graph) {
      MS_LOG(INFO) << "Replace parameter format";
      // The input tensors of heterogeneous graphs or control flow graphs are null.
      // Need to get tensor after ParseControlNodes.
      pynative::GraphAdapter::ReplaceGraphParameterProperties(graph, input_tensors.at(i), device_contexts[i]);
    }
    (void)graph_compiler_->CompileGraphImpl(graph, device_contexts[i]);
    pynative::GraphAdapter::RemoveUnusedValueNodes(graph);
    // PyNative use kernel graph will result in front node and back node is the same; But in pynative task sink, backend
    // still create new kernel graph
    if (root_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph) &&
        !pynative::GraphAdapter::PyNativeEnableTaskSink(root_graph_)) {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, {graph->output()});
    } else {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, graph->front_outputs());
    }
    // Clear front outputs after the outputs is cached.
    graph->set_front_outputs({});
    AnfAlgo::UpdateGraphValidRefPair(graph);
    pynative::GraphAdapter::SensTensorToDevice(graph, device_contexts[i]);
  }

  ParseControlNodes(graph_compiler_info);
  UpdateGraphCompilerInfo(graph_id);
  auto actor_set = runtime::GraphScheduler::GetInstance().Transform(graph_compiler_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  constexpr auto kKernelActorThreshold = 5000;
  // Turning off multithreading may cause stack overflow in control flow scenarios.
  if (no_multi_graph && actor_set->kernel_actors_.size() < kKernelActorThreshold &&
      root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    // Multithreading can cause spikes in memory usage and performance fluctuations.
    actor_set->is_multi_thread_execution_ = false;
    MS_LOG(INFO) << "Actor Multithreading is turned off!";
  }
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  runtime::GraphScheduler::GetInstance().RemoveNodeAddr(graph_compiler_info);

  for (size_t i = 0; i < graphs.size(); ++i) {
    pynative::GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(graphs[i], device_contexts[i]);
    pynative::GraphAdapter::GenerateRefCountForBpropValueNode(graphs[i]);
    graph_adapter_.GenerateBackoffValueNodeOwners(graphs[i]);
  }
  return actor_set;
}

void MSBackend::RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                 const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: begin run actor: " << graph_id;
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto graphs = graph_compiler_info.graphs_;
  auto &device_contexts = graph_compiler_info.device_contexts_;
  if (device_contexts.size() != graphs.size()) {
    MS_LOG(EXCEPTION) << "Graphs size " << graphs.size() << " is not equal to device_contexts size "
                      << device_contexts.size();
  }

  // KernelByKernel: The size of control_nodes is at least 1 since there is return node in the graph.
  // GraphMode: No control nodes.
  bool no_multi_graph = control_nodes_.size() <= 1 && graphs.size() == 1;
  auto actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_id);
  if (actor_set == nullptr) {
    actor_set = RealCompileGraphBeforeRunActor(graph_id, graph_compiler_info, args, no_multi_graph);
    first_step_ = true;
  }
  MS_EXCEPTION_IF_NULL(actor_set);

  if (enable_graph_pipeline_) {
    // 1. Construct stub output.
    MS_EXCEPTION_IF_NULL(root_graph_);
    const auto output_node = root_graph_->output();
    MS_EXCEPTION_IF_NULL(output_node);
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
                                       "MakeStubNode");
    auto stub_output_pair = stub::MakeStubNode(output_node->abstract());
    if (stub_output_pair.second) {
      MS_LOG(DEBUG) << "Enable pynative graph pipeline for actor set: " << graph_id;
      // 2. Async run graph.
      auto &stub_output = stub_output_pair.first;
      MS_EXCEPTION_IF_NULL(stub_output);
      outputs->push_back(stub_output);

      auto run_graph_task = std::make_shared<runtime::RunGraphTask>(
        [=, &graph_compiler_info]() {
          actor_set->output_actor_->SetStubOutput(stub_output);
          RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
        },
        stub_output);
      GilReleaseWithCheck release_gil;
      runtime::Pipeline::Get().frontend_stage()->Push(run_graph_task);
      return;
    }
    enable_graph_pipeline_ = false;
    MS_LOG(INFO)
      << "Failed to create Stub output, encountered an unsupported output type for graph: " << graph_id
      << ". Currently, only output types that include: Tensor, Scalar, String, fixed-length Sequence, are "
         "supported. The single op and graph pipeline has been disabled, so the performance will not be improved.";
  }

  RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
}

void MSBackend::RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set,
                            const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, bool no_multi_graph,
                            VectorRef *outputs) {
  if (!first_step_) {
    WaitTaskFinish();
    WaitMultiStream(graph_compiler_info);
    ContiguousArgs(args, graph_compiler_info);
    WaitTaskFinish();
  } else {
    first_step_ = false;
  }

  auto graphs = graph_compiler_info.graphs_;
  auto &device_contexts = graph_compiler_info.device_contexts_;
  if (root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    for (size_t i = 0; i < graphs.size(); ++i) {
      graph_adapter_.UpdateForwardOutputInBpropGraph(graphs[i], device_contexts[i], no_multi_graph);
      pynative::GraphAdapter::UpdateDynamicValueNodeAbstract(graphs[i]);
    }
  }

  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  // make sure enable input optimize condition right.
  MS_LOG(INFO) << "Start to run graph, args size: " << args.size() << ", graph: " << actor_set->name_;
  runtime::ActorDispatcher::set_enable_sub_graph_execute_for_cur_actor_set(actor_set->enable_kbk_sub_graph_execute_);
  runtime::ActorDispatcher::set_enable_input_optimize_for_cur_actor_set(actor_set->enable_input_optimize_);
  if (!runtime::EnableInputOptimize()) {
    input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (graphs.size() > input_tensors.size()) {
      MS_LOG(EXCEPTION) << "The actor_set " << actor_set->name_ << " graphs size " << graphs.size()
                        << " should less than or equal to inputs size " << input_tensors.size();
    }
    pynative::GraphAdapter::HandleHeterogeneousTensors(input_tensors, device_contexts, actor_set);
    CreateKernelTensor(input_tensors, device_contexts);
    // Release GIL and run actor DAG.
    GilReleaseWithCheck release_gil;
    VectorRef empty_args;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, empty_args);
  } else {
    GilReleaseWithCheck release_gil;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, args);
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  auto output = root_graph_->output();
  MS_LOG(DEBUG) << "Current out " << output->DebugString();
  if (root_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    MS_EXCEPTION_IF_NULL(output_node_);
    root_graph_->set_output(output_node_);
  }
  ConstructOutputs(actor_set, outputs, root_graph_);
  actor_set->output_actor_->FreeSummaryNodeMem();
  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << graph_id;
}

void MSBackend::RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto jit_call_graph = kernel->user_data<pynative::JitCallGraph>();
  MS_EXCEPTION_IF_NULL(jit_call_graph);
  *outputs = jit_call_graph->Run(args);
}

void MSBackend::RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                   VectorRef *outputs) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  ContiguousArgs(args, graph_compiler_info);
  WaitTaskFinish();

  MS_LOG(INFO) << "Status record: begin run graph by single op";
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  const auto &graphs = graph_compiler_info.graphs_;
  auto inputs = GetRunGraphInputs(graph_compiler_info, args);
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    MS_EXCEPTION_IF_NULL(graph);
    std::map<KernelWithIndex, tensor::BaseTensorPtr> op_output_map;
    std::map<AnfNodePtr, size_t> parameter_index;
    GraphOutputInfo graph_output_info;
    graph_output_info.graph_outputs = outputs;
    graph_compiler_->GetParamAndOutputIndex(graph, inputs[graph_index], outputs, &parameter_index,
                                            &graph_output_info.output_indexes);

    std::map<KernelWithIndex, size_t> cnode_ref_count;
    auto iter = cnode_ref_counts_.find(graph->graph_id());
    if (iter == cnode_ref_counts_.end()) {
      graph_compiler_->CalculateRefCount(graph, &cnode_ref_count);
      (void)cnode_ref_counts_.emplace(graph->graph_id(), cnode_ref_count);
    } else {
      cnode_ref_count = iter->second;
    }

    MS_EXCEPTION_IF_NULL(root_graph_);
    if (root_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
      graph_compiler_->CalculateForwardOpOutputCount(graph, inputs[graph_index], &forward_op_output_tensor_id_,
                                                     parameter_index);
      op_backend_.set_forward_tensor_ref_count(forward_op_output_tensor_id_);
    }

    GilReleaseWithCheck gil_release;
    auto is_dynamic = root_graph_->has_flag(kFlagPyNativeBpropGraphIsDynamic);
    bool has_bprop_cut = root_graph_->has_flag(kFlagPyNativeBpropGraphWithBpropCut);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const std::string &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    for (const auto &kernel : graph->execution_order()) {
      MS_EXCEPTION_IF_NULL(kernel);
      MS_LOG(DEBUG) << "Split and run op " << kernel->fullname_with_scope();
      InputInfo input_info;
      VectorRef op_outputs;
      if (has_bprop_cut && common::AnfAlgo::IsBpropCutOpExecInBackend(kernel)) {
        const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;
        RunControlOperator(graph_compiler_, origin_parameters, args, graph, kernel, op_output_map, parameter_index,
                           inputs[graph_index], &input_info, &op_outputs);
        // Execute remaining lazy tasks before PyNative hook exit.
        WaitTaskFinish();
      } else if (common::AnfAlgo::HasNodeAttr(kAttrJitCallNode, kernel)) {
        graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index], false,
                                                 &input_info);
        VectorRef input_args;
        (void)std::transform(input_info.input_values.begin(), input_info.input_values.end(),
                             std::back_inserter(input_args.elements_),
                             [](ValuePtr &value) { return std::move(value); });

        RunMsGradGraph(kernel, input_args, &op_outputs);
        WaitTaskFinish();
      } else {
        const auto &primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
        MS_EXCEPTION_IF_NULL(primitive);
        if (PyBoostAdapter::IsPyBoostRegistered(device_target, primitive->name())) {
          MS_LOG(DEBUG) << "Run " << primitive->name() << " by pyboost";
          graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index], true,
                                                   &input_info);
          runtime::OpRunnerInfo op_runner_info{
            primitive, device_target, input_info.input_values, input_info.input_abs, {}, kernel->abstract()};
          PyBoostAdapter::RunPyBoostCall(&op_runner_info, &op_outputs);
        } else {
          MS_LOG(DEBUG) << "Run " << primitive->name() << " by single op graph";
          session::BackendOpRunInfoPtr op_run_info;
          graph_compiler_->GetSingleOpInputTensors(kernel, op_output_map, parameter_index, inputs[graph_index], false,
                                                   &input_info);
          graph_compiler_->GetSingleOpRunInfoAndGraphInfo(kernel, input_info, is_dynamic, &op_run_info,
                                                          &graph_output_info);
          if (is_dynamic) {
            op_run_info->op_prim = std::make_shared<Primitive>(*op_run_info->op_prim);
            AnfAlgo::SetDynamicAttrToPrim(op_run_info->op_prim);
          }
          op_backend_.Run(op_run_info, device_name_, device_id_, &op_outputs);
        }
      }

      graph_compiler_->UpdateRefCount(input_info.input_kernel, &cnode_ref_count, &op_output_map);

      graph_output_info.graph_output_tensors.clear();
      graph_compiler_->RecoverGraphOutput(kernel, op_outputs, cnode_ref_count, &op_output_map, &graph_output_info);
    }
    WaitTaskFinish();
  }
  python_adapter::PyAdapterCallback::ProcessUnPairedCellHook(true);
  MS_LOG(INFO) << "Status record: end run graph by single op";
}

void MSBackend::RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                    const VectorRef &args, VectorRef *outputs) {
  bool enable_run_graph_by_single_op =
    std::any_of(graph_compiler_info.graphs_.begin(), graph_compiler_info.graphs_.end(),
                [](const KernelGraphPtr &graph) { return graph->has_flag(kFlagEnableRunGraphBySingleOp); });
  if (enable_run_graph_by_single_op) {
    RunGraphBySingleOp(graph_compiler_info, args, outputs);
  } else {
    RunGraphByActors(graph_id, graph_compiler_info, args, outputs);
  }
}

void MSBackend::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::Pipeline::Get().WaitAll();
}

void MSBackend::ClearOpExecutorResource() const { runtime::OpExecutor::GetInstance().Reset(); }

void MSBackend::SyncStream() {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }
}

void MSBackend::ClearResource() {
  graph_compiler_ = std::make_shared<GraphCompiler>();
  graph_id_to_device_context_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  graph_info_to_device_context_.clear();
  control_nodes_.clear();
  actor_to_graph_compiler_info_.clear();
  cnode_ref_counts_.clear();
}

KernelGraphPtr MSBackend::GetGraphById(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  return graph_compiler_->Fetch(graph_id);
}

MS_REGISTER_BACKEND(kMSBackendName, MSBackend)
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
