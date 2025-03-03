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

#ifndef MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_GE_BACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_GE_BACKEND_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include "include/backend/visible.h"
#include "ir/tensor.h"
#include "backend/common/session/kernel_graph_mgr.h"
#include "runtime/hardware/device_context.h"

namespace mindspore::compile {
class BACKEND_EXPORT GEBackend {
 public:
  GEBackend() = default;
  ~GEBackend() = default;
  std::string CompileGraph(const FuncGraphPtr &func_graph, const device::DeviceContext *device_context,
                           const backend::BackendJitConfig &backend_jit_config);

  void RunGraph(const std::string &graph_info, const device::DeviceContext *device_context, const VectorRef &args,
                std::vector<tensor::TensorPtr> *outputs);

  FuncGraphPtr BuildDFGraph(const device::DeviceContext *device_context, const FuncGraphPtr &anf_graph,
                            const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors);
  string ExportDFGraph(const device::DeviceContext *device_context, const std::string &file_name,
                       const FuncGraphPtr &anf_graph, bool is_save_to_file);
  std::unordered_set<std::string> GetInferParameterNames(const device::DeviceContext *device_context);

 private:
  // map<graph_info, KernelGraphPtr>
  mindspore::HashMap<std::string, KernelGraphPtr> graph_map_;
  // if param init in device, for refmode
  mindspore::HashMap<ParameterPtr, bool> is_weight_init_;
  // if weight value update in python, it records the tensor
  static mindspore::HashSet<const tensor::Tensor *> weights_need_reprepare_;
  // graph running step
  mindspore::HashMap<KernelGraphPtr, uint32_t> graph_run_iter_;

  std::string GenerateGraphInfo(GraphId graph_id) { return "kernel_graph_" + std::to_string(graph_id); }
  // for run graph
  void ConstructInputs(const KernelGraphPtr &func_graph, const VectorRef &args,
                       std::vector<tensor::TensorPtr> *inputs_tensor, const device::DeviceContext *device_context);
  void ConstructInputsRefMode(const KernelGraphPtr &func_graph, const VectorRef &args,
                              std::vector<tensor::TensorPtr> *inputs_tensor,
                              const device::DeviceContext *device_context);
  void SetTensorUpdateCallback(const tensor::TensorPtr &update_tensor);
  void SyncTensorData(const tensor::TensorPtr &host_tensor, const std::shared_ptr<device::DeviceAddress> &device_tensor,
                      const AnfNodePtr &node);
  void ConstructOutputs(const KernelGraphPtr &func_graph, std::vector<tensor::TensorPtr> *outputs,
                        const device::DeviceContext *device_contextF);
  bool Copy(const mindspore::device::DeviceAddress *dst_device_tensor,
            const mindspore::device::DeviceAddress *src_device_tensor);
  void UpdateInputsShapeAndSize(const ParameterPtr &input_node,
                                const mindspore::device::DeviceAddressPtr &device_tensor,
                                const tensor::TensorPtr &input_tensor, const device::DeviceContext *device_context);

  // for acl dump
  bool DebugOnStepBegin(const KernelGraphPtr &func_graph);
  bool ACLDump(uint32_t device_id, const KernelGraphPtr &graph);
  void DebugOnStepEnd(const KernelGraphPtr &graph, const device::DeviceContext *device_context, bool dump_flag);

  // for profiling
  bool ProfilerOnStepBegin(const KernelGraphPtr &graph, const device::DeviceContext *device_context);
  void ProfilerOnStepEnd(const device::DeviceContext *device_context, bool profile_started);
};
using GEBackendPtr = std::shared_ptr<GEBackend>;
}  // namespace mindspore::compile
#endif  // MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_GE_BACKEND_H_
