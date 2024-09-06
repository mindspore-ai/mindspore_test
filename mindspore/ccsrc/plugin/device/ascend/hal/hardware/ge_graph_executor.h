/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <set>
#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_summary.h"
#include "plugin/device/ascend/hal/hardware/ge_memory_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace ascend {
struct GeInputData {
  std::vector<GeTensor> ge_inputs;
  std::vector<DeviceAddress *> device_addrs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
};

struct GeOutputData {
  std::vector<GeTensor> ge_outputs;
  std::vector<DeviceAddress *> device_addrs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
};

class GeMessageManager {
 public:
  void SetFeatureMemory(const std::string &name, size_t size) { feature_memorys[name] = size; }
  void SetStream(const std::string &name, size_t size) { streams[name] = size; }
  bool SummaryExist(const std::string &name) const {
    auto iter = summarys.find(name);
    if (iter == summarys.end()) {
      return false;
    }
    return true;
  }
  void SetSummary(const std::string &name, const GraphSummary &summary) { summarys[name] = summary; }
  size_t GetFeatureMemory(const std::string &name) const {
    auto iter = feature_memorys.find(name);
    if (iter == feature_memorys.end()) {
      MS_LOG(EXCEPTION) << "Feature memory " << name << " not found.";
    }
    return iter->second;
  }
  size_t GetStream(const std::string &name) const {
    auto iter = streams.find(name);
    if (iter == streams.end()) {
      MS_LOG(EXCEPTION) << "Stream " << name << " not found.";
    }
    return iter->second;
  }
  GraphSummary GetSummary(const std::string &name) const {
    auto iter = summarys.find(name);
    if (iter == summarys.end()) {
      MS_LOG(EXCEPTION) << "Summary " << name << " not found.";
    }
    return iter->second;
  }

 private:
  HashMap<std::string, size_t> feature_memorys;
  HashMap<std::string, size_t> streams;
  HashMap<std::string, GraphSummary> summarys;
};

class GeGraphExecutor : public GraphExecutor {
 public:
  ~GeGraphExecutor() override = default;
  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) override;
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) override;

  static FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map,
                                   bool export_air);
  void PreprocessBeforeRun(const KernelGraphPtr &graph);
  size_t GetGraphFeatureMemory(const FuncGraphPtr &graph) const override;
  void InitGraphInfo(const FuncGraphPtr &graph) override;

  // For run as kernelmod.
  std::vector<std::pair<uint32_t, uint32_t>> GetGraphRefIndexes(const KernelGraphPtr &graph) const;
  void SetGraphWorkspaceMemory(const KernelGraphPtr &graph, void *device_ptr, size_t size);
  size_t GetGraphWorkSpaceMemory(const std::string &graph_name) const;
  bool CompileGraphForKernel(const KernelGraphPtr &graph);
  bool RunGraphRefModeForKernel(const KernelGraphPtr &graph, const AnfNodeWeakPtr &node,
                                const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                                void *stream);
  void AllocGEFixMemory() const;
  void InitGEFixMemory(const KernelGraphPtr &graph, size_t stream_id) const;

 private:
  bool RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs);
  bool RunGraphRefModeInnner(const FuncGraphPtr &graph, const std::vector<GeTensor> &inputs,
                             std::vector<GeTensor> *outputs, void *stream);
  void BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph);
  void BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph);
  bool CompileGraph(const KernelGraphPtr &graph, const std::map<string, string> &compile_options);
  int64_t CurGraphSinkSize(std::string graph_name);
  std::vector<GeTensor> GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const;
  std::vector<GeTensor> GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const;
  // for GEGraphOp Run
  std::vector<GeTensor> CreateInputGeTensorList(const std::vector<KernelTensor *> &tensorsz,
                                                const KernelGraphPtr &graph);
  std::vector<GeTensor> CreateOutputGeTensorList(const std::vector<KernelTensor *> &tensorsz,
                                                 const KernelGraphPtr &graph);
  GeDeviceResManager *ResManager() const;
  void RunInitGraph(const std::string &graph_name);
  void AddRefCorrespondPairs(const KernelGraphPtr &graph, const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes);
  bool BuildGraph(const KernelGraphPtr &graph, const transform::TensorOrderMap &tensor_order_map);
  DeviceAddressPtr CreateOutputDeviceAddress(const KernelGraphPtr &kernel_graph,
                                             const KernelWithIndex &output_with_index,
                                             size_t need_alloc_output_cnt) const;
  void DoAsyncCkpt(const FuncGraphPtr &graph);
  void SetFlagIgnoreDevicePtr(const FuncGraphPtr &graph);
  mindspore::HashMap<session::KernelGraph *, GeInputData> input_datas_;
  mindspore::HashMap<session::KernelGraph *, GeOutputData> output_datas_;
  // io_index for kernel_graph
  mindspore::HashMap<std::string, std::vector<std::pair<uint32_t, uint32_t>>> io_indexes_;
  std::map<std::string, int64_t> graph_sink_size_;
  int64_t pre_sink_size_{-1};
  bool disable_ge_kernel_ = common::IsDisableRuntimeConfig(common::kRuntimeGeKernel);
  GeMessageManager ge_message_manager_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_
