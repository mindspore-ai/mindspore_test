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

#include "kernel/framework_utils.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/format_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/oplib.h"
#include "ir/graph_utils.h"
#include "kernel/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"
#include "mindapi/base/type_id.h"
#include "mindspore/ccsrc/include/common/debug/common.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
constexpr char kAxis[] = "axis";
constexpr char kKernelObjectTypeNotSupportedStr[] = "KernelObjectTypeNotSupported";

std::string GetCompilerCachePath() { return Common::GetUserDefineCachePath(); }

KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid, kernel_name: " << kernel_name;
    return nullptr;
  }

  std::string kernel_json = bin_map->Search(kernel_name);
  if (!kernel_json.empty()) {
    KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
    // just a tmp solution.
    if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
      MS_LOG(ERROR) << "Read cache json and bin file failed[" << kernel_json << "].";
      return nullptr;
    } else {
      return kernel_pack;
    }
  } else {
    MS_LOG(INFO) << "The cache kernel not found[" << kernel_name << "].";
    return nullptr;
  }
}

KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor) {
  MS_LOG(INFO) << "Insert cache for kernel:" << kernel_name << ", processr:" << processor;
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "Kernel cache is invalid, kernel name :" << kernel_name;
    return nullptr;
  }
  std::string kernel_json = bin_map->kernel_meta_path();
  (void)kernel_json.append(kernel_name).append(kJsonSuffix);
  KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
  if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
    MS_LOG(ERROR) << "Read json and bin file failed[" << kernel_json << "].";
    return nullptr;
  }
  if (bin_map->Insert(kernel_name, kernel_json)) {
    MS_LOG(INFO) << "Kernel insert cache success[" << kernel_json << "], kernel name[" << kernel_name << "].";
  }
  return kernel_pack;
}

void KernelMeta::Initialize(const std::string &backend) {
  auto config_path = GetCompilerCachePath();
  kernel_meta_path_ = config_path + backend + std::string(kKernelMetaSuffix);
  (void)(FileUtils::CreateNotExistDirs(kernel_meta_path_, true));
  initialized_ = true;
}

std::string KernelMeta::Search(const std::string &kernel_name) const {
  if (!initialized_) {
    return "";
  }

  auto iter = kernel_meta_map_.find(kernel_name);
  if (iter == kernel_meta_map_.end()) {
    return "";
  } else {
    return iter->second;
  }
}

bool KernelMeta::Insert(const std::string &kernel_name, const std::string &kernel_json) {
  if (!initialized_) {
    return false;
  }
  kernel_meta_map_[kernel_name] = kernel_json;
  return true;
}

bool SetInputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &inputs, size_t real_input_num,
                               size_t builder_idex, const std::vector<int64_t> &dyn_input_sizes,
                               const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  MS_EXCEPTION_IF_NULL(builder);

  std::vector<TypeId> inputs_device_type;
  std::vector<std::string> inputs_format;
  std::vector<KernelObjectType> inputs_object_type;
  size_t dyn_input_idx = 0;
  size_t kernel_info_index = 0;
  MS_EXCEPTION_IF_NULL(inputs[0]);
  size_t kernel_info_cnt = inputs[0]->dtypes().size();

  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    std::string param_type = input->param_type();
    std::vector<std::string> dtypes = input->dtypes();
    std::vector<std::string> formats = input->formats();
    std::vector<std::string> object_types = input->object_types();
    if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt ||
        object_types.size() != kernel_info_cnt) {
      MS_LOG(DEBUG) << "Set input kernel builder info failed, dtyps size, formats size and object_types size are not "
                       "same. dtypes size: "
                    << dtypes.size() << ", formats size : " << formats.size()
                    << ", object_types size: " << object_types.size();
      return false;
    }

    if (param_type == "dynamic") {
      if (dyn_input_sizes.empty()) {
        MS_LOG(DEBUG) << "Set input kernel builder info failed, dyn_input_sizes's size is 0 when param_type is dynamic";
        return false;
      }

      for (int64_t t = 0; t < dyn_input_sizes[dyn_input_idx]; t++) {
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
        inputs_object_type.push_back(StringToKernelObjectType(object_types[builder_idex]));
      }
    } else if (param_type == "required") {
      kernel_info_index++;
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      inputs_device_type.push_back(type_id);
      inputs_format.push_back(formats[builder_idex]);
      inputs_object_type.push_back(StringToKernelObjectType(object_types[builder_idex]));
    } else {
      if (kernel_info_index < real_input_num) {
        MS_LOG(INFO) << "Set input kernel builder info, input type is optional, input index is :" << kernel_info_index;
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
        inputs_object_type.push_back(StringToKernelObjectType(object_types[builder_idex]));
      }
    }
    dyn_input_idx++;
  }

  builder->SetInputsDeviceType(inputs_device_type);
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsKernelObjectType(inputs_object_type);

  return true;
}

bool SetOutputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &outputs, size_t builder_idex,
                                const size_t &real_output_num,
                                const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  // not now but in the next we need to support dynamic output case
  MS_EXCEPTION_IF_NULL(builder);

  size_t output_idx = 0;
  std::vector<TypeId> outputs_device_type;
  std::vector<std::string> outputs_format;
  std::vector<KernelObjectType> outputs_object_type;
  MS_EXCEPTION_IF_NULL(outputs[0]);
  size_t kernel_info_cnt = outputs[0]->dtypes().size();

  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output_idx >= real_output_num) {
      MS_LOG(DEBUG) << "real_output_num:" << real_output_num << ", output_idx:" << output_idx << " is out of limit!";
      continue;
    }
    size_t output_num = 0;
    if (output->param_type() == "dynamic") {
      if (outputs.size() > 1) {
        MS_EXCEPTION(ArgumentError) << "Dynamic output is unsupported multi output!";
      }
      output_num = real_output_num;
    } else if (output->param_type() == "required") {
      output_num = 1;
    } else {
      if (output_idx < real_output_num) {
        MS_LOG(DEBUG) << "Set output kernel builder info, output type is optional, output index is :" << output_idx;
        output_num = 1;
      }
    }

    for (size_t i = 0; i < output_num; i++) {
      std::vector<std::string> dtypes = output->dtypes();
      std::vector<std::string> formats = output->formats();
      std::vector<std::string> object_types = output->object_types();
      if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt ||
          object_types.size() != kernel_info_cnt) {
        MS_LOG(DEBUG)
          << "Set output kernel builder info failed, dtyps size, formats size and object_types size are not "
             "same. dtypes size: "
          << dtypes.size() << ", formats size : " << formats.size() << ", object_types size: " << object_types.size();
        return false;
      }
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      outputs_device_type.push_back(type_id);
      outputs_format.push_back(formats[builder_idex]);
      outputs_object_type.push_back(StringToKernelObjectType(object_types[builder_idex]));
      output_idx++;
    }
  }

  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_device_type);
  builder->SetOutputsKernelObjectType(outputs_object_type);
  return true;
}

void SetKernelBuildInfo(const std::vector<std::string> &input_formats, const std::vector<TypeId> &input_types,
                        const std::vector<std::string> &output_formats, const std::vector<TypeId> &output_types,
                        const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (kernel_node->kernel_info() == nullptr) {
    kernel_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  }
  if (!kernel_node->kernel_info()->has_build_info()) {
    AnfAlgo::SetSelectKernelBuildInfo(std::make_shared<kernel::KernelBuildInfo>(), kernel_node.get());
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  build_info->SetInputsFormat(input_formats);
  build_info->SetInputsDeviceType(input_types);
  build_info->SetOutputsFormat(output_formats);
  build_info->SetOutputsDeviceType(output_types);
}

void SetKernelBuildInfo(const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder, Processor processor,
                        const std::shared_ptr<const OpInfo> &op_info_ptr) {
  MS_EXCEPTION_IF_NULL(builder);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  builder->SetProcessor(processor);
  auto imply_type = op_info_ptr->imply_type();
  switch (imply_type) {
    case kImplyAKG:
      builder->SetKernelType(AKG_KERNEL);
      break;
    case kImplyTBE:
      builder->SetKernelType(TBE_KERNEL);
      break;
    case kImplyGPU:
      builder->SetKernelType(GPU_KERNEL);
      break;
    case kImplyCPU:
      builder->SetKernelType(CPU_KERNEL);
      break;
    case kImplyAICPU:
      builder->SetKernelType(AICPU_KERNEL);
      break;
    case kImplyBISHENG:
      builder->SetKernelType(BISHENG_KERNEL);
      break;
    default:
      MS_LOG(EXCEPTION) << "Unknown Imply Type.";
      break;
  }
}

bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr, Processor processor,
                   std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  size_t real_input_num = AnfAlgo::GetInputElementNum(kernel_node);
  size_t real_output_num = AnfAlgo::GetOutputElementNum(kernel_node);
  std::vector<std::shared_ptr<OpIOInfo>> inputs = op_info_ptr->inputs_ptr();
  std::vector<std::shared_ptr<OpIOInfo>> outputs = op_info_ptr->outputs_ptr();
  std::vector<int64_t> dyn_input_sizes;
  auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (primitive->GetAttr("dyn_input_sizes") != nullptr) {
    dyn_input_sizes = GetValue<std::vector<int64_t>>(primitive->GetAttr("dyn_input_sizes"));
  }
  if (dyn_input_sizes.empty() && inputs.size() < real_input_num) {
    MS_LOG(WARNING) << "The size of inputs in OpIOInfo should be great than real input. Inputs size in OpIOInfo:"
                    << inputs.size() << ", real input num: " << real_input_num
                    << ", node: " << kernel_node->fullname_with_scope();
    return false;
  }
  if (inputs.size() > 0) {
    if (inputs[0] == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Inputs[0] is nullptr. Op name: " << op_name;
    }
    size_t kernel_info_cnt = inputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetInputKernelBuilderInfo(inputs, real_input_num, j, dyn_input_sizes, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set inputs kernel builder info failed. Op name: " << op_name;
        return false;
      }

      if (outputs.size() > 0) {
        if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
          MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed. Op name: " << op_name;
          return false;
        }
      }

      kernel_info_list->push_back(builder->Build());
    }
  } else if (outputs.size() > 0) {
    if (outputs[0] == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Outputs[0] is nullptr. Op name: " << op_name;
    }
    size_t kernel_info_cnt = outputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed. Op name: " << op_name;
        return false;
      }

      kernel_info_list->push_back(builder->Build());
    }
  } else {
    if (processor == AICPU) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);
      kernel_info_list->push_back(builder->Build());
    }
  }
  return true;
}

void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path) {
  std::string path = base_path + json_name + kInfoSuffix;
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream filewrite(realpath.value());
  if (!filewrite.is_open()) {
    MS_LOG(ERROR) << "Open file '" << realpath.value() << "' failed!";
    return;
  }
  filewrite << info << std::endl;
  filewrite.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
}

std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                          const std::vector<AnfNodePtr> &input_list,
                                                          const std::vector<AnfNodePtr> &output_list) {
  std::vector<std::pair<AnfNodePtr, size_t>> output_index;
  for (size_t i = 0; i < output_list.size(); ++i) {
    auto const &output = output_list[i];
    MS_EXCEPTION_IF_NULL(output);
    bool found = false;
    auto pree_node = common::AnfAlgo::VisitKernel(output, 0);
    auto pos = std::find(std::begin(node_list), std::end(node_list), pree_node.first);
    if (pos != std::end(node_list)) {
      output_index.push_back(pree_node);
      continue;
    }
    auto ret = std::find(std::begin(input_list), std::end(input_list), pree_node.first);
    if (ret != std::end(input_list)) {
      output_index.push_back(std::make_pair(pree_node.first, 0));
      found = true;
    }
    if (!found) {
      MS_EXCEPTION(ArgumentError) << "Output [" << i << "][" << output->DebugString(AnfNode::DebugStringLevel::kLevel2)
                                  << "] of [" << output->func_graph()->ToString() << "] found no related kernel info.";
    }
  }
  return output_index;
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_lists = TopoSort(func_graph->get_return());
  for (auto const &node : node_lists) {
    if (!AnfUtils::IsRealKernel(node) || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex))) {
      node_list->push_back(node);
    }
  }
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(input_list);

  GetValidKernelNodes(func_graph, node_list);

  auto parameters = func_graph->parameters();
  (void)input_list->insert(input_list->cbegin(), parameters.begin(), parameters.end());

  GetFuncGraphOutputNodes(func_graph, output_list);
}

void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(output_list);
  auto func_output = func_graph->output();
  MS_EXCEPTION_IF_NULL(func_output);
  if (func_output->isa<CNode>()) {
    // multi output.
    auto cnode = func_output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(kAnfPrimitiveIndex);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      for (size_t input_idx = 1; input_idx < cnode->size(); ++input_idx) {
        auto input_node = cnode->input(input_idx);
        MS_EXCEPTION_IF_NULL(input_node);
        if (input_node->isa<CNode>() && common::AnfAlgo::GetInputTensorNum(input_node) == 0) {
          continue;
        }
        output_list->push_back(common::AnfAlgo::VisitKernel(input_node, 0).first);
      }
    } else {
      // single output.
      output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
    }
  } else {
    // single output.
    output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
  }
}

std::vector<int64_t> GetReduceAttrAxis(const CNodePtr &cnode) {
  if (common::AnfAlgo::GetInputTensorNum(cnode) != 1 || AnfAlgo::GetOutputElementNum(cnode) != 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "The reduce node [" << cnode->DebugString()
                               << "] is not single input or single output." << trace::DumpSourceLines(cnode);
  }
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis_attr = primitive->GetAttr(kAxis);
  if (axis_attr == nullptr) {
    MS_LOG(ERROR) << "This node doesn't have axis attr. Node info [" << cnode->DebugString() << "]";
    return {};
  }
  std::vector<int64_t> axis_list;
  if (axis_attr->isa<Int64Imm>()) {
    (void)axis_list.emplace_back(GetValue<int64_t>(axis_attr));
  } else {
    axis_list = GetValue<std::vector<int64_t>>(axis_attr);
  }
  return axis_list;
}

Processor GetProcessorFromContext() {
  kernel::Processor processor = kernel::Processor::UNKNOWN;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_info = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_info == kGPUDevice) {
    processor = kernel::Processor::CUDA;
  } else if (device_info == kAscendDevice) {
    processor = kernel::Processor::AICORE;
  } else if (device_info == kCPUDevice) {
    processor = kernel::Processor::CPU;
  }
  return processor;
}

std::string GetStrProcessorFromContext() {
  auto processor = GetProcessorFromContext();
  string str_processor = kernel::kProcessorUnknown;
  if (processor == kernel::Processor::CUDA) {
    str_processor = kernel::kProcessorCuda;
  } else if (processor == kernel::Processor::AICORE) {
    str_processor = kernel::kProcessorAiCore;
  } else if (processor == kernel::Processor::CPU) {
    str_processor = kernel::kProcessorCpu;
  }
  return str_processor;
}

bool GetShapeSize(const ShapeVector &shape, const TypePtr &type_ptr, int64_t *size_i) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  size_t type_byte = GetTypeByte(type_ptr);
  if (type_byte == 0) {
    return false;
  }
  for (size_t j = 0; j < shape.size(); j++) {
    if (shape[j] <= 0) {
      MS_LOG(DEBUG) << "shape[" << shape << "] has invalid value(less equal 0), set size to 0";
      size_i[0] = 0;
      return true;
    }
    size_i[0] = LongMulWithOverflowCheck(size_i[0], shape[j]);
  }
  size_i[0] = LongMulWithOverflowCheck(size_i[0], SizeToInt(type_byte));
  return true;
}

bool IsDynamicParamKernel(const std::string &op_name) {
  const auto &op_info = kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU);
  constexpr auto kParamDynamic = "dynamic";

  if (op_info == nullptr) {
    return false;
  }

  const auto &input_io_info = op_info->inputs_ptr();
  if (input_io_info.size() != 1 || input_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  const auto &output_io_info = op_info->outputs_ptr();
  if (output_io_info.size() != 1 || output_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  return true;
}

std::pair<std::string, ExceptionType> KernelObjectTypeNotSupportWarning(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto GetObjectTypeStr = [](const std::vector<TypeId> &object_types) {
    std::vector<std::string> object_type_strs;
    (void)std::transform(object_types.begin(), object_types.end(), std::back_inserter(object_type_strs), TypeIdLabel);
    return std::accumulate(object_type_strs.begin(), object_type_strs.end(), std::string(),
                           [](const std::string &x, const std::string &y) { return x.empty() ? y : x + ", " + y; });
  };
  const std::string warn_str = std::string(kKernelObjectTypeNotSupportedStr) + ": unsupported kernel object type for " +
                               kernel_node->fullname_with_scope() + " with inputs (" +
                               GetObjectTypeStr(AnfAlgo::GetAllInputObjectType(kernel_node)) + "), outputs (" +
                               GetObjectTypeStr(AnfAlgo::GetAllOutputObjectType(kernel_node)) + ").";
  return {warn_str, TypeError};
}

bool IsKernelObjectTypeNotSupportedError(const std::string &error_str) {
  return error_str.find(kKernelObjectTypeNotSupportedStr) != std::string::npos;
}

KernelObjectType StringToKernelObjectType(const std::string &object_type) {
  static const std::unordered_map<std::string, KernelObjectType> object_type_maps = {
    {"unknown", KernelObjectType::UNKNOWN_TYPE},
    {"tensor", KernelObjectType::TENSOR},
    {"scalar", KernelObjectType::SCALAR},
    {"tuple", KernelObjectType::TUPLE},
    {"tuple_unfold", KernelObjectType::TUPLE_UNFOLD},
  };
  auto iter = object_type_maps.find(object_type);
  if (iter == object_type_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal input object type: " << object_type;
  }
  return iter->second;
}

void UnfoldKernelBuildInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  auto input_num = kernel_build_info->GetInputNum();
  auto output_num = kernel_build_info->GetOutputNum();
  if (input_num == 0 && output_num == 0) {
    return;
  }
  const auto &input_kernel_object_types = kernel_build_info->GetAllInputKernelObjectTypes();
  const auto &output_kernel_object_types = kernel_build_info->GetAllOutputKernelObjectTypes();
  const auto &input_dtypes = kernel_build_info->GetAllInputDeviceTypes();
  const auto &output_dtypes = kernel_build_info->GetAllOutputDeviceTypes();
  const auto &input_formats = kernel_build_info->GetAllInputFormats();
  const auto &output_formats = kernel_build_info->GetAllOutputFormats();

  std::vector<TypeId> unfold_input_dtypes;
  std::vector<TypeId> unfold_output_dtypes;
  std::vector<std::string> unfold_input_formats;
  std::vector<std::string> unfold_output_formats;
  auto Append = [&unfold_input_dtypes, &unfold_input_formats, &unfold_output_dtypes, &unfold_output_formats,
                 input_dtypes, input_formats, output_dtypes, output_formats, input_num,
                 output_num](bool in_or_out, size_t index) {
    if (in_or_out) {
      MS_EXCEPTION_IF_CHECK_FAIL((input_num > index), "Input index is out of range.");
      unfold_input_dtypes.push_back(input_dtypes[index]);
      unfold_input_formats.push_back(input_formats[index]);
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL((output_num > index), "Output index is out of range.");
      unfold_output_dtypes.push_back(output_dtypes[index]);
      unfold_output_formats.push_back(output_formats[index]);
    }
  };
  auto RepeatAppend = [Append](bool in_or_out, size_t index, size_t times) {
    while (times > 0) {
      Append(in_or_out, index);
      times--;
    }
  };

  for (size_t i = 0; i < input_kernel_object_types.size(); ++i) {
    if (input_kernel_object_types[i] == kernel::KernelObjectType::TUPLE_UNFOLD) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel_node, i);
      auto unfold_num = GetOutputNum(input_node);
      MS_LOG(DEBUG) << kernel_node->fullname_with_scope() << " input idnex:" << i << " unfold num:" << unfold_num;
      RepeatAppend(true, i, unfold_num);
    } else {
      Append(true, i);
    }
  }

  for (size_t i = 0; i < output_kernel_object_types.size(); ++i) {
    if (output_kernel_object_types[i] == kernel::KernelObjectType::TUPLE_UNFOLD) {
      auto unfold_num = GetOutputNum(kernel_node);
      MS_LOG(DEBUG) << kernel_node->fullname_with_scope() << " output idnex:" << i << " unfold num:" << unfold_num;
      // Multiple outputs are expanded in the kernel attr(For example BatchNorm op).
      if (output_num == unfold_num) {
        for (size_t j = 0; j < unfold_num; ++j) {
          Append(false, j);
        }
      } else {
        RepeatAppend(false, i, unfold_num);
      }
    } else {
      Append(false, i);
    }
  }

  SetKernelBuildInfo(unfold_input_formats, unfold_input_dtypes, unfold_output_formats, unfold_output_dtypes,
                     kernel_node);
}

int64_t CalOutputTupleSize(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  bool is_bprop_cut = common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimBpropCut);
  bool skip = (is_bprop_cut && node->abstract()->isa<abstract::AbstractSparseTensor>());
  if (skip || !common::AnfAlgo::IsTupleOutput(node)) {
    return -1;
  }
  const auto &real_node = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem}).first;
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(real_node);
  if (build_info != nullptr) {
    auto output_object = AnfAlgo::GetOutputKernelObjectType(real_node, 0);
    if (output_object != kernel::KernelObjectType::TUPLE_UNFOLD) {
      return -1;
    }
  }
  auto output_size = static_cast<int64_t>(AnfAlgo::GetOutputElementNum(node));
  if (node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    output_size = 0;
    auto make_tuple = node->cast<CNodePtr>();
    size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(make_tuple);
    for (size_t j = 0; j < tuple_input_num; ++j) {
      // using for graph kernel
      auto dyn_input_node = common::AnfAlgo::GetInputNode(make_tuple, j);
      // Handle tuple nested scenes.
      if (dyn_input_node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(dyn_input_node, prim::kPrimMakeTuple)) {
        output_size += CalOutputTupleSize(dyn_input_node);
      } else {
        output_size++;
      }
    }
  }
  return output_size == 0 ? -1 : output_size;
}

void SetDynamicInputSizeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
      common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimPartial)) {
    return;
  }
  std::vector<int64_t> dyn_input_sizes;
  auto input_obj_types = AnfAlgo::GetInputKernelObjectTypes(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    if (i < input_obj_types.size() && input_obj_types[i] == kernel::KernelObjectType::TUPLE_UNFOLD) {
      auto input_node = common::AnfAlgo::GetInputNode(cnode, i);
      dyn_input_sizes.push_back(CalOutputTupleSize(input_node));
    } else {
      dyn_input_sizes.push_back(-1);
    }
  }
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode);
  }
}

// In compile stage, run resize when kernel is not dynamic shape or has no value depend list.
bool CheckResizeCondition(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->input(0));
  if (!AnfAlgo::NodeValueIsFuncGraph(node->input(0))) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(DEBUG) << "Skip resize for " << node->DebugString() << ", for reason is dynamic shape";
      return false;
    }
    if (common::AnfAlgo::IsDynamicValue(node)) {
      MS_LOG(DEBUG) << "Skip resize for " << node->DebugString() << ", for reason is dynamic value";
      return false;
    }
  }

  return true;
}

// If node is not aclnn or is a Reshape, need convert input into contiguous.
bool NeedCheckInputContiguous(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_type = AnfAlgo::GetKernelType(cnode);
  if (kernel_type != KernelType::OPAPI_KERNEL && kernel_type != KernelType::INTERNAL_KERNEL) {
    return true;
  }
  return false;
}
}  // namespace kernel
}  // namespace mindspore
