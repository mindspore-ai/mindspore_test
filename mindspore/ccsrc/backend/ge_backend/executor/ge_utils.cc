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

#include "backend/ge_backend/executor/ge_utils.h"

#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <nlohmann/json.hpp>
#include "include/common/utils/anfalgo.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/graph_ir/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/kernel_graph.h"
#include "abstract/abstract_value.h"
#include "utils/phase.h"
#include "plugin/res_manager/ascend/hccl_adapter/hccl_adapter.h"
#include "plugin/res_manager/ascend/device_context_conf/op_tuning_conf.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};
constexpr uint32_t kDumpModeAicoreOverflow = 1;
constexpr uint32_t kDumpModeAtomicOverflow = 2;
constexpr uint32_t kDumpModeAll = 3;

std::string ShapesToString(const ShapeArray &shapes) {
  std::stringstream buffer;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      buffer << ",";
    }
    buffer << "[";
    const auto &shape = shapes[i];
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j != 0) {
        buffer << ",";
      }
      buffer << shape[j];
    }
    buffer << "]";
  }
  return buffer.str();
}

template <typename Map, typename K = typename Map::key_type, typename V = typename Map::mapped_type>
std::string MapToString(const Map &value) {
  std::stringstream buffer;
  buffer << "{";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it != value.begin()) {
      buffer << ", ";
    }
    buffer << it->first << ": " << it->second;
  }
  buffer << "}";
  return buffer.str();
}

inline std::string GetPhasePrefix() {
  const std::string &phase = PhaseManager::GetInstance().phase();
  auto pos = phase.find('.');
  if (pos != std::string::npos) {
    return phase.substr(0, pos);
  }

  return "";
}

void UpdateTopoOrderOptions(const string &graph_name, OptionMap *option) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  const auto &topo_order = context->get_param<std::string>(MS_CTX_TOPO_ORDER);
  if (topo_order.empty()) {
    return;
  }

  nlohmann::json topo_order_json = nlohmann::json::parse(topo_order);
  auto topo_order_iter = topo_order_json.find(graph_name);
  if (topo_order_iter == topo_order_json.end()) {
    return;
  }
  MS_LOG(INFO) << "Update topo order for graph " << graph_name << " to " << topo_order_iter.value();
  std::string topo_sorting_mode = "1";
  if (topo_order_iter.value() == "bfs") {
    topo_sorting_mode = "0";
  } else if (topo_order_iter.value() == "dfs") {
    topo_sorting_mode = "1";
  } else if (topo_order_iter.value() == "rdfs") {
    topo_sorting_mode = "2";
  }
  (*option)["ge.topoSortingMode"] = topo_sorting_mode;
}

OptionMap GetComputeGraphOptions(const ShapeArray &input_shapes, bool is_dynamic_shape) {
  OptionMap options{};
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto max_threshold = ms_context->get_param<std::string>(MS_CTX_HOST_SCHEDULING_MAX_THRESHOLD);
  if (!max_threshold.empty()) {
    (void)options.emplace("ge.exec.hostSchedulingMaxThreshold", max_threshold);
  }
  if (!is_dynamic_shape) {
    return options;
  }
  (void)options.emplace("ge.exec.dynamicGraphExecuteMode", "dynamic_execute");
  (void)options.emplace("ge.exec.dataInputsShapeRange", ShapesToString(input_shapes));
  return options;
}

void GetComputeGraphReuseOptions(const FuncGraphPtr &graph, OptionMap *option) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(option);
  auto enable_io_reuse = common::GetEnv("MS_ENABLE_IO_REUSE");
  MS_LOG(INFO) << "Enable io reuse: " << enable_io_reuse;
  if (enable_io_reuse != "1") {
    return;
  }
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  if (!outputs.empty()) {
    std::string value;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto output = outputs[i];
      const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
      auto &output_node = output_with_index.first;
      MS_EXCEPTION_IF_NULL(output_node);
      // Parameter and value can not been reused.
      if (output_node->isa<Parameter>() || output_node->isa<ValueNode>()) {
        MS_LOG(INFO) << "Output is parameter or value node, not support reuse, index is: " << i;
        continue;
      }
      (void)value.append(std::to_string(i));
      (void)value.append(",");
    }
    if (!value.empty()) {
      value.pop_back();
      MS_LOG(INFO) << "key: ge.exec.outputReuseMemIndexes, value: " << value << ",Graph name: " << graph->ToString();
      (void)option->insert(std::make_pair("ge.exec.outputReuseMemIndexes", value));
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (graph->has_flag(backend::ge_backend::kGraphFlagHasGetNext) &&
      !graph->has_flag(backend::ge_backend::kGraphNeedIteration)) {
    MS_LOG(INFO) << "key: ge.exec.inputReuseMemIndexes, value: 0."
                 << ", Graph name: " << graph->ToString();
    (void)option->insert(std::make_pair("ge.exec.inputReuseMemIndexes", "0"));
  }
}

// ge.exec.allow_hf32 default value is "10"(enable Conv, disable Matmul) set by CANN
void SetAscendHF32Config(const std::shared_ptr<MsContext> &ms_context_ptr,
                         std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  std::string allow_matmul_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_MATMUL_ALLOW_HF32);
  std::string allow_conv_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_CONV_ALLOW_HF32);
  if (allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    MS_LOG(INFO) << "The default value of allow_matmul_hf32 and allow_conv_hf32 are set by CANN.";
  } else if (allow_matmul_hf32.empty() && !allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + std::string("0");
  } else if (!allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = std::string("1") + allow_matmul_hf32;
  } else {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + allow_matmul_hf32;
  }

  MS_LOG(INFO) << "allow_matmul_hf32: " << allow_matmul_hf32 << ", allow_conv_hf32: " << allow_conv_hf32;
}

void SetAscendConfig(const std::shared_ptr<MsContext> &ms_context_ptr, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  std::string topo_sorting_mode = "0";
  if (ms_context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    topo_sorting_mode = "2";
  }
  (*ge_options)["ge.topoSortingMode"] = topo_sorting_mode;
  // disable RemoveSameConstPass, it will be caused the communication failed on multi-card.
  (*ge_options)["ge.disableOptimizations"] = "RemoveSameConstPass";

  (*ge_options)["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  MS_LOG(INFO) << "Set GE topo mode to memory-priority.";

  (*ge_options)["ge.exec.staticMemoryPolicy"] = "2";
  MS_LOG(INFO) << "Set staticMemoryPolicy to default mode 2.";

  if (ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) != "") {
    (*ge_options)["ge.jit_compile"] = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
    MS_LOG(INFO) << "Set jit_compile " << ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) << ".";
  } else {
    (*ge_options)["ge.jit_compile"] = "2";
    MS_LOG(INFO) << "The default value of jit_compile is set to 2.";
  }

  auto ge_exception_dump = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_EXCEPTION_DUMP);
  (*ge_options)["ge.exec.enable_exception_dump"] = ge_exception_dump;

  SetAscendHF32Config(ms_context_ptr, ge_options);

  if (ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) != "") {
    (*ge_options)["ge.exec.op_precision_mode"] = ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
    MS_LOG(INFO) << "Set op_precision_mode " << ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) << ".";
  }
}

void SetHcclOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(inst_context);
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_table_file = common::GetEnv("MINDSPORE_HCCL_CONFIG_PATH");
  if (env_table_file.empty()) {
    env_table_file = common::GetEnv("RANK_TABLE_FILE");
  }
  auto simulation_level = common::GetEnv(kSimulationLevel);
  if (!simulation_level.empty()) {
    env_table_file = "";
  }
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  auto enable_hccl = inst_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
  auto escluster_config_path = common::GetEnv("ESCLUSTER_CONFIG_PATH");

  MS_LOG(INFO) << "Values for hccl options: env_table_file[" << env_table_file << "], simulation_level["
               << simulation_level << "], env_rank_id[" << env_rank_id << "], env_device_id[" << env_device_id
               << "], enable_hccl[" << enable_hccl << "], UseDynamicCluster[" << common::UseDynamicCluster() << "].";
  if (enable_hccl &&
      (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty()) ||
       hccl::HcclAdapter::GetInstance().UseHcclCM()) &&
      !(common::UseDynamicCluster() && !env_table_file.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    } else if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      hccl::HcclAdapter::AddCMEnvToHcclOption(ge_options);
    }

    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
  } else if (!escluster_config_path.empty()) {
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID] all be set in ENV.";
  }
}

}  // namespace

bool IsGeTrain() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_ge = context->backend_policy() == "ge";
  bool enable_training = GetPhasePrefix() == "train";
  if (enable_ge && enable_training) {
    return true;
  }
  return false;
}

std::string GetGraphName(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  return graph->ToString();
}

void SetPassthroughGeOptions(std::string option_level, OptionMap *options) {
  const auto &new_options = AnfAlgo::GetGeOptions(option_level);
  for (auto &[key, value] : new_options) {
    (*options)[key] = value;
    MS_LOG(INFO) << "Set ge " << option_level << " option: {" << key << ", " << value << "}";
  }
}

bool AddFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = backend::ge_backend::NewConverter(anf_graph, GetPhasePrefix());
  backend::ge_backend::GenFakeGraph(anf_graph->ToString(), converter);
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  ShapeArray shape_array;
  bool dynamic_shape_inputs = false;
  auto options = GetComputeGraphOptions(shape_array, dynamic_shape_inputs);
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)backend::ge_backend::AddGraph(graph_name, backend::ge_backend::GetComputeGraph(converter));
  (void)backend::ge_backend::AddGraph(init_graph, backend::ge_backend::GetInitGraph(converter));
  (void)backend::ge_backend::AddGraph(BROADCAST_GRAPH_NAME, backend::ge_backend::GetBroadcastGraph(converter));

  return true;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const backend::ge_backend::TensorOrderMap &init_inputs_map,
                bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = backend::ge_backend::NewConverter(anf_graph, GetPhasePrefix());
  bool is_cloud = true;
  bool need_aoe = false;
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    backend::ge_backend::SetTraining(converter, false);
    backend::ge_backend::SetExportAir(converter, true);
    is_cloud = false;
  }
  backend::ge_backend::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  backend::ge_backend::GenerateBroadcastGraph(converter, init_inputs_map);
  backend::ge_backend::GenerateCheckpointGraph(converter);
  auto err_code = backend::ge_backend::ErrCode(converter);
  if (err_code != 0) {
    backend::ge_backend::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }

  if (device::ascend::OpTuningConf::GetInstance()->EnableAoeOnline()) {
    need_aoe = true;
  }
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  auto options = GetComputeGraphOptions(converter->input_shapes(), converter->dynamic_shape_inputs());
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)backend::ge_backend::AddGraph(graph_name, backend::ge_backend::GetComputeGraph(converter),
                                      backend::ge_backend::DfGraphConfig(options, is_cloud, need_aoe, export_air));
  (void)backend::ge_backend::AddGraph(init_graph, converter->GetInitGraph());
  (void)backend::ge_backend::AddGraph(BROADCAST_GRAPH_NAME, backend::ge_backend::GetBroadcastGraph(converter));
  return true;
}

void GetGeSessionOptions(backend::ge_backend::SessionOptions *options) {
  MS_EXCEPTION_IF_NULL(options);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  (*options)["ge.enablePrintOpPass"] = "0";
  (*options)["ge.constLifecycle"] = "graph";
  (*options)["ge.exec.formatMode"] = "0";
  auto format_mode = common::GetEnv("MS_FORMAT_MODE");
  if (format_mode == "1" || (format_mode.empty() && ms_context->ascend_soc_version() != "ascend910")) {
    MS_LOG(INFO) << "Set GE option ge.exec.formatMode to 1.";
    (*options)["ge.exec.formatMode"] = "1";
  }

  // options from set_context
  if (ms_context->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.graphMemoryMaxSize"] = ms_context->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.variableMemoryMaxSize"] = ms_context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

  auto atomic_clean_policy = ms_context->get_param<std::string>(MS_CTX_ATOMIC_CLEAN_POLICY);
  if (atomic_clean_policy.empty()) {
    atomic_clean_policy = "1";
  }
  (*options)["ge.exec.atomicCleanPolicy"] = atomic_clean_policy;
  MS_LOG(INFO) << "Set GE atomic clean policy to " << atomic_clean_policy << ".";
  (*options)["ge.graphRunMode"] = "1";
}

void GetGeGlobalOptions(std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto ms_context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context_ptr);

  SetHcclOptions(ms_context_ptr, ge_options);
  (*ge_options)["ge.exec.jobId"] = "0";
  MS_LOG(INFO) << "Set ge.exec.jobId to default value 0";

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(INFO) << "Got empty proto lib path, cannot set ge.opsProtoLibPath.";
  }

  SetAscendConfig(ms_context_ptr, ge_options);

  auto op_debug_level = common::GetEnv("MS_COMPILER_OP_LEVEL");
  if (!op_debug_level.empty()) {
    (*ge_options)["ge.opDebugLevel"] = op_debug_level;
    MS_LOG(INFO) << "Use MS_COMPILER_OP_LEVEL, op debug level:" << op_debug_level;
  }

  // Enable the global variable acc may cause accuracy problems in train+eval
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }

  // enable overflow detection
  (*ge_options)["ge.exec.overflow"] = "1";
  // enable deterministic
  (*ge_options)[::ge::DETERMINISTIC] = ms_context_ptr->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? "1" : "0";
  MS_LOG(INFO) << "Set ge::DETERMINISTIC to " << (*ge_options)[::ge::DETERMINISTIC];
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
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
