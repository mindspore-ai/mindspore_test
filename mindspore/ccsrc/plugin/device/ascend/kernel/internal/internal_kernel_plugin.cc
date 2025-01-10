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

#include "plugin/device/ascend/kernel/internal/internal_kernel_plugin.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/internal_helper.h"
#include "plugin/device/ascend/kernel/internal/pyboost/acme_kernel_info.h"
#include "plugin/device/ascend/kernel/internal/pyboost/acme_pyboost_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/internal_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/factory/ms_factory.h"
#include "kernel/framework_utils.h"
#include "op_def/math_op_name.h"
#include "op_def/nn_op_name.h"
#include "acl/acl_base.h"
#include "transform/acl_ir/acl_helper.h"
#include "utils/phase.h"
#include "utils/ms_context.h"

namespace mindspore::kernel {
namespace {
constexpr auto kPhaseNameDecode = "decode";
constexpr auto kPhaseNameIncrement = "increment";
constexpr auto kQuantLinearSparseName = "QuantLinearSparse";
constexpr auto kQuantBatchMatmulName = "QuantBatchMatmul";
constexpr auto CONST_2 = 2;
constexpr auto Align16 = 16;
constexpr auto kQuantLinearSparseBiasIdx = 5;  // primitive input weight deq_scale compress_idx bias
constexpr auto kMatMulWeightIdx = 2;           // primitive input weight ...

// unordered_map vector<vector<vector<size_t>>> represents:
// list[op_name][0] for phase prefill, list[op_name][1] for phase increment;
// list[op_name][][0] for input indices, list[op_name][][0] for output indices.
static std::unordered_map<std::string, std::vector<std::vector<std::vector<size_t>>>> kNzFormatOpsList = {
  {kMatMulOpName, {{{0, 1}, {}}, {{1}, {}}}},
  {kQuantLinearSparseName, {{{0}, {}}, {{}, {}}}},
  {kQuantBatchMatmulName, {{{0, 1}, {}}, {{1}, {}}}},
  {kPagedAttentionOpName, {{{0, 1, 2, 7}, {0}}, {{0, 1, 2, 7}, {0}}}},
  {kFlashAttentionScoreOpName, {{{0, 1, 2, 6}, {3}}, {{0, 1, 2, 6}, {3}}}},
  {kReshapeAndCacheOpName, {{{2, 3}, {}}, {{2, 3}, {}}}}};

// unordered_map mean:
// key is input_idx, value is special_format value
// ATTENTION_INPUT_QKV: ms_nd_shape{b, s, h} need convert to {b * s, h}, then transform nz format
// ATTENTION_INPUT_MASK: ms_nd_shape{b, 1, s, s} need convert to {b, 1, s, s}, then transform nz format
static const std::unordered_map<std::string, std::unordered_map<size_t, int64_t>> kSpecialNzFormatOpsList = {
  {kPagedAttentionOpName, {{0, internal::TransDataParam::ATTENTION_INPUT_QKV}}},
  {kFlashAttentionScoreOpName,
   {{0, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {1, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {2, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {6, internal::TransDataParam::ATTENTION_INPUT_MASK}}}};

int64_t GetSpecialFormat(const AnfNodePtr &cur_node, const AnfNodePtr &input_node, const size_t input_idx) {
  MS_EXCEPTION_IF_NULL(cur_node);
  MS_EXCEPTION_IF_NULL(input_node);
  int64_t special_format_input = internal::TransDataParam::NORMAL;

  // cur cnode has special format input
  auto special_format_iter = kSpecialNzFormatOpsList.find(AnfUtils::GetCNodeName(cur_node));
  if (special_format_iter != kSpecialNzFormatOpsList.end()) {
    auto iter = special_format_iter->second.find(input_idx);
    if (iter != special_format_iter->second.end()) {
      special_format_input = iter->second;
    } else {
      special_format_input = internal::TransDataParam::NORMAL;
    }
  } else if (input_node->isa<CNode>()) {
    // input cnode has special format output: pa & fa output format is nz
    auto special_iter = kSpecialNzFormatOpsList.find(AnfUtils::GetCNodeName(input_node));
    if (special_iter != kSpecialNzFormatOpsList.end()) {
      special_format_input = internal::TransDataParam::ATTENTION_INPUT_QKV;
    }
  }
  return special_format_input;
}

bool IsKernelGraphOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &outputs = common::AnfAlgo::GetAllOutputIndexByReturnTypes(func_graph->output());
  return std::find_if(outputs.begin(), outputs.end(), [&node](const auto &output) {
           const auto &real_pair = common::AnfAlgo::VisitKernelWithReturnType(node, 0);
           return output.first == node || (real_pair.first == output.first && real_pair.second == output.second);
         }) != outputs.end();
}

bool IsNeedInsertTransDataForGraphOut(const AnfNodePtr &node, const std::vector<std::string> &output_formats) {
  // output is graph output & format is nz
  if (IsKernelGraphOutput(node) &&
      std::any_of(output_formats.begin(), output_formats.end(),
                  [](const std::string &format) { return !transform::AclHelper::CheckDefaultSupportFormat(format); })) {
    return true;
  }
  return false;
}

bool NeedSetParameterFormat(const AnfNodePtr &input_node, const std::string &new_format,
                            const std::string &input_format) {
  std::string old_format = input_format;
  if (transform::AclHelper::CheckDefaultSupportFormat(old_format) &&
      !transform::AclHelper::CheckDefaultSupportFormat(new_format)) {
    transform::SetParameterFormat(input_node, new_format, &old_format);
    if (old_format != input_format) {
      return true;
    }
  }
  return false;
}

void GetMsTypesList(const CNodePtr &kernel, std::vector<TypeId> *ms_in_dtypes, std::vector<TypeId> *ms_out_dtypes) {
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel);

  for (size_t i = 0; i < input_num; i++) {
    auto cur_input_type = mindspore::device::ascend::GetInputDeviceType(kernel, i);
    if (mindspore::device::ascend::IsEmptyTupleInput(kernel, i, cur_input_type)) {
      cur_input_type = TypeId::kNumberTypeInt64;
    }
    (void)ms_in_dtypes->push_back(cur_input_type);
  }

  for (size_t i = 0; i < output_num; i++) {
    (void)ms_out_dtypes->push_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
  }
  return;
}

void UpdateNzFormatOpsList(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfUtils::GetCNodeName(node) == prim::kPrimGroupedMatmul->name() &&
      common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
    auto dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
    if (!dyn_input_sizes.empty()) {
      auto weight_num = static_cast<size_t>(dyn_input_sizes[0]);
      std::vector<size_t> input_idx;
      for (size_t i = weight_num; i < weight_num * 2; ++i) {
        input_idx.emplace_back(i);
      }
      kNzFormatOpsList[prim::kPrimGroupedMatmul->name()] = {{input_idx, {}}, {input_idx, {}}};
    }
  }
}
}  // namespace

KernelModPtr InternalKernelPlugin::BuildKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string op_fullname = anf_node->fullname_with_scope();
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  // Easy to compare accuracy and performance, later changed to debug
  KernelModPtr kernel_ptr;
  if (Factory<InternalKernelMod>::Instance().IsRegistered(opname)) {
    MS_LOG(INFO) << "Supported by InternalKernel: " << opname;
    kernel_ptr = std::static_pointer_cast<KernelMod>(Factory<InternalKernelMod>::Instance().Create(opname));
  }

  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "internal can't find Kernel[" << opname << "]";
    return nullptr;
  }
  kernel_ptr->set_fullname(op_fullname);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!kernel_ptr->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "#dmsg#Kernel build failed:#dmsg#Initialize internal kernel op["
                                          << anf_node->fullname_with_scope() << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#internal kernel op[" << cnode->fullname_with_scope()
                        << "] Resize failed.";
    }
  }

  return kernel_ptr;
}

bool InternalKernelPlugin::IsRegisteredKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  std::vector<TypeId> ms_in_dtypes;
  std::vector<TypeId> ms_out_dtypes;
  GetMsTypesList(cnode, &ms_in_dtypes, &ms_out_dtypes);
  if (Factory<InternalKernelMod>::Instance().IsRegistered(opname)) {
    auto internal_op_name = TransInternalOpName(opname);
    auto internal_in_dtypes = InternalKernelModInOutMap::GetInstance()->MapInternalInputDtypes(opname, ms_in_dtypes);
    auto internal_out_dtypes = InternalKernelModInOutMap::GetInstance()->MapInternalOutputDtypes(opname, ms_out_dtypes);
    return internal::IsInternalKernelDtypesSupported(internal_op_name, internal_in_dtypes, internal_out_dtypes);
  }

  return false;
}

bool CheckMatMulWeightIsUnAlign(const AnfNodePtr &node) {
  const auto op_name = AnfUtils::GetCNodeName(node);
  if (op_name == kMatMulOpName || op_name == kQuantLinearSparseName || op_name == kQuantBatchMatmulName) {
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    // In QuantLinearSparse op, Shape of Weight is compressed, check unalign by bias.
    auto idx = (op_name == kQuantLinearSparseName) ? kQuantLinearSparseBiasIdx : kMatMulWeightIdx;
    auto base_shape_ptr = inputs[idx]->Shape();  // check weight
    MS_EXCEPTION_IF_NULL(base_shape_ptr);
    auto shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto data_shape = shape_ptr->shape();
    MS_LOG(INFO) << "data_shape = " << data_shape;
    auto len = data_shape.size();
    if ((len >= 1 && data_shape[len - 1] % Align16 != 0) ||
        (len >= CONST_2 && data_shape[len - CONST_2] % Align16 != 0)) {
      return true;
    }
  }
  return false;
}

bool IsDecodePhase(const std::string &phase) {
  return phase.rfind(kPhaseNameDecode) != std::string::npos || phase.rfind(kPhaseNameIncrement) != std::string::npos;
}

bool CheckOpSupprtNzFormatOnly(const bool &enable_internal_op, const std::string &op_name) {
  return !enable_internal_op &&
         (op_name == kMatMulOpName || op_name == kQuantLinearSparseName || op_name == kQuantBatchMatmulName);
}

void InternalKernelPlugin::GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node,
                                                                     std::vector<std::string> *input_formats,
                                                                     std::vector<std::string> *output_formats) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  UpdateNzFormatOpsList(node);

  auto phase = PhaseManager::GetInstance().phase();
  auto phase_idx = static_cast<size_t>(IsDecodePhase(phase));
  auto op_name = AnfUtils::GetCNodeName(node);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_op = (std::find(enable_op_list.begin(), enable_op_list.end(), op_name) != enable_op_list.end());
  auto support_nz_format_only = CheckOpSupprtNzFormatOnly(enable_op, op_name);

  auto format_idx_iter = kNzFormatOpsList.find(op_name);
  if (format_idx_iter != kNzFormatOpsList.end()) {
    auto input_nz_format_idx = format_idx_iter->second[phase_idx][0];
    auto output_nz_format_idx = format_idx_iter->second[phase_idx][1];
    if (CheckMatMulWeightIsUnAlign(node) || support_nz_format_only) {
      input_nz_format_idx.push_back(0);
      output_nz_format_idx.push_back(0);
    }

    for (const auto &input_idx : input_nz_format_idx) {
      input_formats->at(input_idx) = kOpFormat_FRAC_NZ;
    }
    for (const auto &output_idx : output_nz_format_idx) {
      output_formats->at(output_idx) = kOpFormat_FRAC_NZ;
    }
  }

  std::vector<size_t> special_inputs;
  std::vector<int64_t> special_format_inputs;
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto first_node = kernel_with_index.first;
    if (first_node->isa<ValueNode>()) {
      auto value_node = first_node->cast<ValueNodePtr>();
      auto value = value_node->value();
      if (value->isa<None>()) {
        continue;
      }
    }
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    input_format = NeedSetParameterFormat(kernel_with_index.first, input_formats->at(i), input_format)
                     ? input_formats->at(i)
                     : input_format;
    // for reshapeext input_idx == 1, do not insert transdata
    if (AnfUtils::GetCNodeName(node) == kReshapeExtOpName && i == 1) {
      continue;
    }
    if ((!transform::AclHelper::CheckDefaultSupportFormat(input_format) ||
         !transform::AclHelper::CheckDefaultSupportFormat(input_formats->at(i))) &&
        input_format != input_formats->at(i)) {
      (void)special_inputs.emplace_back(i);
      (void)special_format_inputs.emplace_back(GetSpecialFormat(node, kernel_with_index.first, i));
    }
  }
  if (!special_inputs.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
    if (std::any_of(special_format_inputs.begin(), special_format_inputs.end(),
                    [](const int64_t format_type) { return format_type != internal::TransDataParam::NORMAL; })) {
      common::AnfAlgo::SetNodeAttr(kAttrInternalSepcialFormat, MakeValue(special_format_inputs), node);
    }
  }
  // if graph output is nz format need insert transdata
  if (IsNeedInsertTransDataForGraphOut(node, *output_formats)) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialFormat, MakeValue(true), node);
  }
}

void InternalKernelPlugin::AcmeKernelCall(const std::shared_ptr<pyboost::OpRunner> &op,
                                          const ValuePtrList input_values) {
  std::shared_ptr<AcmeKernelInfo> kernel_info = nullptr;
  const std::string kernelname = op->primitive()->name();
  if (Factory<AcmeKernelInfo>::Instance().IsRegistered(kernelname)) {
    MS_LOG(INFO) << "Supported by Acme Op: " << kernelname;
    kernel_info = std::static_pointer_cast<AcmeKernelInfo>(Factory<AcmeKernelInfo>::Instance().Create(kernelname));
  }
  if (kernel_info == nullptr) {
    MS_LOG(WARNING) << "Acme can't find op[" << kernelname << "]";
    return;
  }

  kernel_info->Call(op, input_values);
}

MS_KERNEL_PLUGIN_FACTORY_REG(InternalKernelPlugin, InternalKernelPlugin);
}  // namespace mindspore::kernel
