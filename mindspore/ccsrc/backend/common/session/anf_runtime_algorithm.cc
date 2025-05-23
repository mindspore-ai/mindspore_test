/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "include/backend/anf_runtime_algorithm.h"

#include <memory>
#include <algorithm>
#include <map>
#include <set>
#include <functional>
#include "mindspore/ops/op_def/ascend_op_name.h"
#include "mindspore/ops/op_def/math_op_name.h"
#include "mindspore/ops/op_def/lite_op_name.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ir/anf.h"
#include "utils/log_adapter.h"
#include "ir/func_graph_cloner.h"
#include "utils/shape_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/backend/kernel_info.h"
#include "common/device_address.h"
#include "include/backend/optimizer/helper.h"
#include "common/kernel.h"
#include "common/kernel_build_info.h"
#include "common/common_utils.h"
#include "include/common/utils/ms_device_shape_transfer.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "abstract/ops/primitive_infer_map.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/trace_base.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "ir/tensor_py_wrapperbase.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::session {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
using device::KernelInfo;
using kernel::KernelBuildInfoPtr;
using kernel::KernelMod;
using kernel::KernelModPtr;
constexpr char kDisableKernelBackoff[] = "MS_DISABLE_KERNEL_BACKOFF";
constexpr char kMindsporeDumpConfig[] = "MINDSPORE_DUMP_CONFIG";

namespace {
constexpr size_t kReturnDataIndex = 1;
constexpr size_t kSwitchTrueBranchIndex = 2;
constexpr auto kPatternUnknown = "";

std::string PrintKernelFormatAndType(const std::string &fmt, const TypeId &type, const std::vector<int64_t> &shape) {
  std::ostringstream buffer;
  buffer << "<" << TypeIdLabel(type);
  if (!fmt.empty()) {
    buffer << "x" << fmt << shape;
  }
  buffer << ">";
  return buffer.str();
}

[[maybe_unused]] struct AnfDumpHandlerRegister {
  AnfDumpHandlerRegister() {
    AnfDumpHandler::SetPrintInputTypeShapeFormatHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      std::ostringstream buffer;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      for (size_t i = 0; i < input_num; ++i) {
        if (i != 0) {
          buffer << ", ";
        }
        auto format = AnfAlgo::GetInputFormat(node, i);
        auto type = AnfAlgo::GetInputDeviceDataType(node, i);
        auto shape = AnfAlgo::GetInputDeviceShape(node, i);
        buffer << PrintKernelFormatAndType(format, type, shape);
      }
      return buffer.str();
    });
    AnfDumpHandler::SetPrintOutputTypeShapeFormatHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      std::ostringstream buffer;
      size_t output_num = AnfAlgo::GetOutputTensorNum(node);
      for (size_t i = 0; i < output_num; ++i) {
        if (i != 0) {
          buffer << ", ";
        }
        auto format = AnfAlgo::GetOutputFormat(node, (node->isa<Parameter>() ? 0 : i));
        auto type = AnfAlgo::GetOutputDeviceDataType(node, (node->isa<Parameter>() ? 0 : i));
        auto shape = AnfAlgo::GetOutputDeviceShape(node, (node->isa<Parameter>() ? 0 : i));
        buffer << PrintKernelFormatAndType(format, type, shape);
      }
      return buffer.str();
    });
    AnfDumpHandler::SetPrintInputKernelObjectTypesHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      auto input_obj_types = AnfAlgo::GetInputKernelObjectTypes(node);
      return std::accumulate(
        input_obj_types.begin(), input_obj_types.end(), std::string(), [](std::string &a, const KernelObjectType &b) {
          return a.empty() ? kernel::KernelObjectTypeLabel(b) : a + ", " + kernel::KernelObjectTypeLabel(b);
        });
    });
    AnfDumpHandler::SetPrintOutputKernelObjectTypesHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      auto output_obj_types = AnfAlgo::GetOutputKernelObjectTypes(node);
      return std::accumulate(
        output_obj_types.begin(), output_obj_types.end(), std::string(), [](std::string &a, const KernelObjectType &b) {
          return a.empty() ? kernel::KernelObjectTypeLabel(b) : a + ", " + kernel::KernelObjectTypeLabel(b);
        });
    });
  }
} callback_register;

tensor::TensorPtr GetForwardOutputTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      // If output used as sens, output will create(clone) a fake tensor with device address is nullptr for memory
      // usage. It has is_forward_output flag, which will be used for tensor input mask, and affect single op graph
      // cache.
      if (tensor->is_forward_output() && tensor->device_address() != nullptr) {
        return tensor;
      }
    }
  }
  return nullptr;
}

size_t GetOutputTensorNumByKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->kernel_info());
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetAllOutputDeviceTypes().size();
}

bool ContainScalarOut(const AbstractBasePtr &abs) {
  // Check the output abstract of node whether is scalar.
  if ((abs != nullptr) && (abs->isa<abstract::AbstractScalar>())) {
    return true;
  }
  // Check the output abstracts of node whether have scalar.
  if ((abs != nullptr) && (abs->isa<abstract::AbstractSequence>())) {
    auto abs_seq = abs->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->dynamic_len()) {
      const auto &element_abs = abs_seq->dynamic_len_element_abs();
      return (element_abs == nullptr) || (element_abs->isa<abstract::AbstractScalar>());
    }
    const auto &elements = abs_seq->elements();
    bool has_scalar_out = std::any_of(elements.begin(), elements.end(),
                                      [](const AbstractBasePtr &element) { return ContainScalarOut(element); });
    return has_scalar_out;
  }
  return false;
}

void TensorValueToVector(const ValuePtr &value, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(outputs);
  if (value->isa<ValueSequence>()) {
    auto value_tuple = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        outputs->emplace_back(tensor);
      } else if (element->isa<Scalar>()) {
        auto scalar = element->cast<ScalarPtr>();
        MS_EXCEPTION_IF_NULL(scalar);
        outputs->emplace_back(ScalarToTensor(scalar));
      } else if (element->isa<ValueSequence>()) {
        VectorRef tuple;
        TensorValueToVector(element, &tuple);
        outputs->emplace_back(tuple);
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    outputs->emplace_back(tensor);
  } else if (value->isa<Scalar>()) {
    auto scalar = value->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(scalar);
    outputs->emplace_back(ScalarToTensor(scalar));
  }
}
}  // namespace

AnfNodePtr AnfRuntimeAlgorithm::MakeMonadValueNode(const KernelGraphPtr &kg) {
  MS_EXCEPTION_IF_NULL(kg);
  return kg->NewValueNode(kUMonad->ToAbstract(), kUMonad);
}

// Convert: a = former(xxx)
//          b = latter(x, xxx)
// To:      a = former(xxx)
//          d1 = Depend(x, a)
//          b = latter(d1, xxx)
//          ...
//          out = Depend(out, latter)
void AnfRuntimeAlgorithm::KeepOrder(const KernelGraphPtr &kg, const AnfNodePtr &former, const AnfNodePtr &latter) {
  MS_EXCEPTION_IF_NULL(kg);
  MS_EXCEPTION_IF_NULL(former);
  MS_EXCEPTION_IF_NULL(latter);
  if (latter->isa<CNode>()) {
    auto latter_cnode = latter->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(latter_cnode);
    constexpr size_t inputsize = 2;
    constexpr size_t kFirstDataInputIndex = 1;
    if (latter_cnode->size() < inputsize) {
      return;
    }
    auto latter_input = latter_cnode->input(kFirstDataInputIndex);
    auto depend1 = kg->NewCNode({NewValueNode(prim::kPrimDepend), latter_input, former});
    MS_EXCEPTION_IF_NULL(depend1);
    MS_EXCEPTION_IF_NULL(latter_input);
    depend1->set_abstract(latter_input->abstract());
    latter_cnode->set_input(kFirstDataInputIndex, depend1);

    auto return_node = kg->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    auto depend2 = kg->NewCNode(
      {NewValueNode(prim::kPrimDepend), return_node->cast<CNodePtr>()->input(kFirstDataInputIndex), latter});
    MS_EXCEPTION_IF_NULL(depend2);
    depend2->set_abstract(return_node->cast<CNodePtr>()->input(kFirstDataInputIndex)->abstract());
    kg->set_output(depend2);
    MS_LOG(DEBUG) << "former: " << former->DebugString() << ", latter: " << latter->DebugString()
                  << ", depend1: " << depend1->DebugString() << ", depend2: " << depend2->DebugString();
  }
}

size_t AnfRuntimeAlgorithm::GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t res;
  TypePtr type = node->Type();
  if (type == nullptr) {
    res = 0;
  } else if (type->isa<Tuple>() || type->isa<List>()) {
    const auto &kernel_info = node->kernel_info();
    if (kernel_info == nullptr || (!kernel_info->has_build_info())) {
      return 1;
    }
    res = GetOutputTensorNumByKernelInfo(node);
  } else if (type->isa<TypeNone>()) {
    res = 0;
  } else if (type->isa<CSRTensorType>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    res = kCSRTensorOutputNum;
  } else if (type->isa<COOTensorType>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    res = kCOOTensorOutputNum;
  } else if (AnfUtils::NeedJumpMonadOutput(node) && type->isa<MonadType>()) {
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    res = 0;
  } else {
    res = 1;
  }
  return res;
}

size_t AnfRuntimeAlgorithm::GetOutputNumWithoutKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &kernel_info = node->kernel_info();
  if (kernel_info != nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Kernel info is not null for node:" << node->DebugString();
  }

  size_t res;
  TypePtr type = node->Type();
  if (type == nullptr) {
    res = 0;
  } else if (type->isa<Tuple>() || type->isa<List>()) {
    res = 1;
  } else if (type->isa<TypeNone>()) {
    res = 0;
  } else if (type->isa<CSRTensorType>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    res = kCSRTensorOutputNum;
  } else if (type->isa<COOTensorType>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    res = kCOOTensorOutputNum;
  } else if (AnfUtils::NeedJumpMonadOutput(node) && type->isa<MonadType>()) {
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    res = 0;
  } else {
    res = 1;
  }
  return res;
}

namespace {
bool IsTupleHasDynamicSequence(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if (sequence_abs->dynamic_len() || sequence_abs->dynamic_len_element_abs() != nullptr) {
    return true;
  }
  if (std::any_of(sequence_abs->elements().begin(), sequence_abs->elements().end(),
                  [](const abstract::AbstractBasePtr &abs) { return IsTupleHasDynamicSequence(abs); })) {
    return true;
  }
  return false;
}
}  // namespace

size_t AnfRuntimeAlgorithm::GetOutputElementNum(const AnfNodePtr &node) {
  if (node->abstract() != nullptr && IsTupleHasDynamicSequence(node->abstract())) {
    return common::AnfAlgo::GetOutputNumByAbstract(node->abstract());
  }
  return AnfUtils::GetOutputTensorNum(node);
}

size_t GetOutputTensorMemSizeImpl(const AnfNodePtr &node, size_t output_index, const ShapeVector &real_shape) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= AnfAlgo::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << AnfAlgo::GetOutputTensorNum(node) << "] of node!";
  }
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto shape = real_shape;
  auto format = AnfAlgo::GetOutputFormat(node, output_index);
  auto dtype = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (shape.empty() && format != kOpFormat_DEFAULT) {
    shape = trans::PaddingShape(shape, format, AnfAlgo::GetOutputReshapeType(node, output_index), node);
    shape = trans::TransShapeToDevice(shape, format, node, output_index, dtype);
  }
  // scalar's output shape is a empty vector
  size_t tensor_size = type_size * SizeOf(shape);
  return tensor_size;
}

size_t AnfRuntimeAlgorithm::GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index,
                                                   const ShapeVector &real_shape) {
  if (IsDynamic(real_shape)) {
    MS_LOG(EXCEPTION) << "The shape is " << real_shape << " dynamic shape , can not get OutputTensorMemSize";
  }
  return GetOutputTensorMemSizeImpl(node, output_index, real_shape);
}

size_t AnfRuntimeAlgorithm::GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  auto shape = AnfAlgo::GetOutputDeviceShape(node, output_index);
  if (IsDynamic(shape)) {
    auto max_shape = common::AnfAlgo::GetOutputMaxShape(node, output_index);
    if (!max_shape.empty()) {
      shape = max_shape;
      MS_LOG(DEBUG) << "shape[" << shape << "] is dynamic, using max_shape[" << max_shape << "] instead.";
    } else {
      shape = {1};
      MS_LOG(DEBUG) << "shape[" << shape << "] is dynamic, set default to {1}";
    }
  }
  return GetOutputTensorMemSizeImpl(node, output_index, shape);
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllOutputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Not real kernel:"
                                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllOutputFormats();
  return format;
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllInputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Not real kernel:"
                                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllInputFormats();
  return format;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllInputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Not real kernel:"
                                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllInputDeviceTypes();
  return types;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Not real kernel:"
                                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllOutputDeviceTypes();
  return types;
}

std::string AnfRuntimeAlgorithm::GetOriginDataFormat(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Not real kernel:"
                                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return kOpFormat_DEFAULT;
  }
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return kOpFormat_DEFAULT;
  }
  auto format = build_info->GetOriginDataFormat();
  return format;
}

std::string AnfRuntimeAlgorithm::GetOutputFormat(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > AnfAlgo::GetOutputElementNum(node) && (!common::AnfAlgo::IsDynamicSequence(node))) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output index:" << output_idx
                                      << " is out of the node output range :" << AnfAlgo::GetOutputElementNum(node)
                                      << " #node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  if (common::AnfAlgo::CheckAbsSparseTensor(node)) {
    return kOpFormat_DEFAULT;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return AnfAlgo::GetPrevNodeOutputFormat(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  std::string format;
  // If the output is TUPLE, output format list's size is 1. So we use the first element as the output format.
  // This scenario could happen before 'insert_type_transform_op' pass.
  auto output_obj_types = build_info->GetAllOutputKernelObjectTypes();
  if (!output_obj_types.empty() && output_obj_types[kIndex0] == KernelObjectType::TUPLE) {
    MS_LOG(DEBUG) << "TUPLE only has one output. So use index 0 output format.";
    format = build_info->GetOutputFormat(kIndex0);
  } else {
    format = build_info->GetOutputFormat(output_idx);
  }
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Node [" << node->DebugString() << "]"
                                      << " has a invalid output format" << trace::DumpSourceLines(node);
  }
  return format;
}

std::string AnfRuntimeAlgorithm::GetInputFormat(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Input index :" << input_idx << " is out of the number node Input range :"
                                      << common::AnfAlgo::GetInputTensorNum(node) << "#node [" << node->DebugString()
                                      << "]" << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputFormat(node, input_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetInputFormat(input_idx);
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Node [" << node->DebugString() << "]"
                                      << " input index:" << input_idx << " has a invalid input format\n"
                                      << trace::DumpSourceLines(node);
  }
  return format;
}

bool AnfRuntimeAlgorithm::IsEquivalentFormat(const Format &src_format, const Format &dst_format) {
  if (src_format == dst_format) {
    return true;
  }

  // Equivalent default format.
  if (((src_format == DEFAULT_FORMAT) || (src_format == NCHW) || (src_format == ND)) &&
      ((dst_format == DEFAULT_FORMAT) || (dst_format == NCHW) || (dst_format == ND))) {
    return true;
  }

  return false;
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return GetOutputReshapeType(kernel_with_index.first, kernel_with_index.second);
}

std::vector<KernelObjectType> AnfRuntimeAlgorithm::GetInputKernelObjectTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Empty build info for node:" << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString();
  }
  return build_info->GetAllInputKernelObjectTypes();
}

KernelObjectType AnfRuntimeAlgorithm::GetInputKernelObjectType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Empty build info for node:" << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString();
  }
  const auto &input_kernel_obj_types = build_info->GetAllInputKernelObjectTypes();
  if (input_idx >= input_kernel_obj_types.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Input index " << input_idx
                                      << ", but the node input kernel object types size just "
                                      << input_kernel_obj_types.size() << ". node: " << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString() << "." << trace::DumpSourceLines(node);
  }
  return input_kernel_obj_types[input_idx];
}

std::vector<KernelObjectType> AnfRuntimeAlgorithm::GetOutputKernelObjectTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return {};
  }
  return build_info->GetAllOutputKernelObjectTypes();
}

KernelObjectType AnfRuntimeAlgorithm::GetOutputKernelObjectType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Empty build info for node:" << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString();
  }
  const auto &output_kernel_obj_types = build_info->GetAllOutputKernelObjectTypes();
  if (output_idx >= output_kernel_obj_types.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output index " << output_idx
                                      << ", but the node output kernel object types size just "
                                      << output_kernel_obj_types.size() << ". node: " << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString() << "." << trace::DumpSourceLines(node);
  }
  return output_kernel_obj_types[output_idx];
}

std::vector<KernelObjectType> AnfRuntimeAlgorithm::GetOutputElementsKernelObjectTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Empty build info for node:" << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString();
  }
  return build_info->GetAllOutputElementsKernelObjectTypes();
}

bool AnfRuntimeAlgorithm::GetValid(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Empty build info for node:" << node->fullname_with_scope()
                                      << ", debug name:" << node->DebugString();
  }
  return build_info->valid();
}

bool AnfRuntimeAlgorithm::IsRealSquenceOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<KernelObjectType> objects = GetOutputKernelObjectTypes(node);
  bool is_real_tuple = false;
  if (objects.empty()) {
    return false;
  } else {
    is_real_tuple = (objects[0] == KernelObjectType::TUPLE);
  }
  return is_real_tuple;
}

bool AnfRuntimeAlgorithm::IsShapesDynamic(const std::vector<ShapeVector> &shapes) {
  return std::any_of(shapes.cbegin(), shapes.cend(), [](const auto &shape) { return IsDynamic(shape); });
}

ShapeVector AnfRuntimeAlgorithm::GetRuntimePaddingShape(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  ShapeVector host_shape;
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    // Scalar has no shape.
    if (node_value->isa<Scalar>()) {
      return {};
    }
    if (node_value->isa<StringImm>()) {
      auto string_value = node_value->cast<StringImmPtr>();
      MS_EXCEPTION_IF_NULL(string_value);
      return {SizeToLong(string_value->ToString().size())};
    }
    if (node_value->isa<ValueSequence>()) {
      MS_LOG(INFO) << "GetRuntimePaddingShape does not support the value sequence for value node:"
                   << node->fullname_with_scope() << ", debug name:" << node->DebugString();
      return {0};
    }
    auto tensor = node_value->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << " The node[ " << node->DebugString() << "]'s cannot convert ";
    }
    host_shape = tensor->shape();
  } else {
    host_shape = common::AnfAlgo::GetOutputInferShape(node, index);
  }
  auto format = GetOutputFormat(node, index);
  if (trans::IsNeedPadding(format, host_shape)) {
    host_shape = trans::PaddingShape(host_shape, format, GetOutputReshapeType(node, index), node);
  }
  return host_shape;
}

ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx) {
  auto format = GetOutputFormat(node, output_idx);
  auto infer_shape = common::AnfAlgo::GetOutputInferShape(node, output_idx, IsRealSquenceOutput(node));
  if (infer_shape.empty()) {
    return infer_shape;
  }

  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape)) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetOutputReshapeType(node, output_idx), node);
  }
  auto dtype = GetOutputDeviceDataType(node, output_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, output_idx, dtype);
}

ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx,
                                                      ShapeVector real_shape) {
  auto format = GetOutputFormat(node, output_idx);
  if (real_shape.empty()) {
    return real_shape;
  }

  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, real_shape)) {
    real_shape = trans::PaddingShape(real_shape, format, GetOutputReshapeType(node, output_idx), node);
  }
  auto dtype = GetOutputDeviceDataType(node, output_idx);
  return trans::TransShapeToDevice(real_shape, format, node, output_idx, dtype);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputDeviceShape(const AnfNodePtr &node, size_t input_idx) {
  auto format = GetInputFormat(node, input_idx);
  auto infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, input_idx);
  if (infer_shape.empty()) {
    return infer_shape;
  }
  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape)) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetInputReshapeType(node, input_idx), node);
  }
  auto dtype = GetInputDeviceDataType(node, input_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, input_idx, dtype, false);
}

std::string AnfRuntimeAlgorithm::GetInputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "The index:" << input_idx << " is out of range of the node's input size : "
                                      << common::AnfAlgo::GetInputTensorNum(node) << "#node[" << node->DebugString()
                                      << "]" << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, input_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsInputDefaultPadding()) {
    return "";
  }
  return build_info->GetInputReshapeType(input_idx);
}

std::string AnfRuntimeAlgorithm::GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsOutputDefaultPadding()) {
    return "";
  }
  return build_info->GetOutputReshapeType(output_idx);
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllInputReshapeType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsInputDefaultPadding()) {
    return {};
  }
  return build_info->GetAllInputReshapeType();
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllOutputReshapeType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsOutputDefaultPadding()) {
    return {};
  }
  return build_info->GetAllOutputReshapeType();
}

TypeId AnfRuntimeAlgorithm::GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > AnfAlgo::GetOutputElementNum(node)) {
    if (common::AnfAlgo::IsDynamicSequence(node)) {
      auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto build_info = kernel_info->select_kernel_build_info();
      MS_EXCEPTION_IF_NULL(build_info);
      return build_info->GetOutputDeviceType(0);
    }
    MS_LOG_WITH_NODE(EXCEPTION, node) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                                      << AnfAlgo::GetOutputElementNum(node) << "#node [ " << node->DebugString() << "]"
                                      << trace::DumpSourceLines(node);
  }
  if (common::AnfAlgo::CheckAbsSparseTensor(node)) {
    return common::AnfAlgo::GetSparseTypeIdAt(node, output_idx);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);

  // If node has only one output and it is Tuple, in build_info, it only has one same dtype, so set output_dix as zero.
  if (build_info->GetOutputNum() == 1 && build_info->GetOutputKernelObjectType(0) == kernel::KernelObjectType::TUPLE) {
    output_idx = 0;
  }

  auto dtype = build_info->GetOutputDeviceType(output_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Node [" << node->DebugString() << "] has a invalid dtype"
                                      << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "The index [" << input_idx << "] is out of range of the node's input size [ "
                                      << common::AnfAlgo::GetInputTensorNum(node) << "#node [ " << node->DebugString()
                                      << "]" << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, 0);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto dtype = build_info->GetInputDeviceType(input_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Node [" << node->DebugString() << "]"
                                      << " has a invalid dtype." << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputDeviceDataType(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
}

// get output device addr of anf_node
const DeviceAddress *AnfRuntimeAlgorithm::GetOutputAddr(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto tensor = GetForwardOutputTensor(node);
  if (tensor != nullptr) {
    return dynamic_cast<const DeviceAddress *>(tensor->device_address().get());
  }

  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(cnode, 0);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output_idx " << output_idx << " of node " << node->DebugString()
                                      << " output addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx,
                                                           bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto tensor = GetForwardOutputTensor(node);
  if (tensor != nullptr) {
    return std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
  }

  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(cnode, 0);
  }
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output_idx " << output_idx << " of node " << node->DebugString()
                                      << " node:" << node << " output addr is not exist."
                                      << trace::DumpSourceLines(node);
  }
  return addr;
}

// get output device addr of anf_node
bool AnfRuntimeAlgorithm::OutputAddrExist(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() > 1) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
      return OutputAddrExist(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
    }
    return false;
  }
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info_ptr = node->kernel_info();
  if (kernel_info_ptr == nullptr) {
    return false;
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_info_ptr);
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->OutputAddrExist(output_idx);
}

bool AnfRuntimeAlgorithm::WorkspaceAddrExist(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->WorkspaceAddrExist(output_idx);
}

const DeviceAddress *AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                bool skip_nop_node) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputAddr(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                   bool skip_nop_node) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
}

std::tuple<abstract::BaseShapePtr, TypePtr, ValuePtr> AnfRuntimeAlgorithm::GetAbstractInfo(const AnfNodePtr &node,
                                                                                           size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  abstract::BaseShapePtr shape;
  TypePtr type;
  ValuePtr value;

  // Create output kernel tensor if not exists.
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    value = value_node->value();
    auto abs = node->abstract();
    if (abs == nullptr) {
      MS_EXCEPTION_IF_NULL(value);
      abs = value->ToAbstract();
      value_node->set_abstract(abs);
    }
    MS_EXCEPTION_IF_NULL(abs);
    shape = abs->GetShape();
    type = abs->GetType();
  } else {
    const auto &abs = AnfAlgo::GetNodeAbstractByIndex(node, output_idx);
    MS_EXCEPTION_IF_NULL(abs);
    shape = abs->GetShape();
    type = abs->GetType();
    value = nullptr;
  }

  // Insert cast pass will change the device type for some reason like CPU do not support fp16 actually,
  // so the output infer type and device type will be different, we change the output tensor to the real device type.
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<TensorType>()) {
    auto real_device_type = AnfAlgo::GetOutputDeviceDataType(node, output_idx);
    auto abs_tensor_type = type->Clone()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(abs_tensor_type);
    auto abs_element = abs_tensor_type->element();
    if (abs_element != nullptr) {
      auto abs_tensor_element_type = abs_element->type_id();
      if (real_device_type != kTypeUnknown && real_device_type != abs_tensor_element_type) {
        MS_LOG(INFO) << "For kernel " << node->DebugString() << ", the infer type of output[" << output_idx << "] is "
                     << TypeIdToString(abs_tensor_element_type) << ", but the device type is "
                     << TypeIdToString(real_device_type)
                     << ". Maybe there has insert cast pass which changed the device type."
                     << " So we change the tensor type from " << TypeIdToString(abs_tensor_element_type) << " to "
                     << TypeIdToString(real_device_type);
        abs_tensor_type->set_element(TypeIdToType(real_device_type));
        // Use new tensor type with device data type.
        type = abs_tensor_type;
      }
    }
  }

  return std::make_tuple(shape, type, value);
}

bool AnfRuntimeAlgorithm::ExistOutputKernelTensor(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    MS_LOG(EXCEPTION) << "Kernel info is null for node:" << node->DebugString() << ", node address:" << node << " .";
  }
  return kernel_info->OutputAddrExist(output_idx) || kernel_info->OutputKernelTensorExist(output_idx);
}

const KernelTensorPtr &AnfRuntimeAlgorithm::GetOutputKernelTensor(const AnfNodePtr &node, size_t output_idx,
                                                                  bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return AnfRuntimeAlgorithm::GetPrevNodeOutputKernelTensor(cnode, 0, skip_nop_node);
  }

  // Get output kernel tensor if exists.
  if (kernel_info->OutputKernelTensorExist(output_idx)) {
    return kernel_info->GetOutputKernelTensor(output_idx);
  }

  MS_LOG_WITH_NODE(EXCEPTION, node) << "Can not find kernel tensor for node : " << node->DebugString()
                                    << ", output index: " << output_idx;
}

const KernelTensorPtr &AnfRuntimeAlgorithm::GetOrCreateOutputKernelTensor(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);

  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get kernel info for node:" << node->DebugString() << " index:" << output_idx;
  }

  // Get output kernel tensor if exists.
  if (kernel_info->OutputKernelTensorExist(output_idx)) {
    return kernel_info->GetOutputKernelTensor(output_idx);
  }

  auto [shape, type, value] = GetAbstractInfo(node, output_idx);
  auto kernel_tensor = std::make_shared<KernelTensor>(shape, type, value);
  // Handle the format diff between host and device, need set format before Resize KernelMod.
  kernel_tensor->SetStringFormat(GetOutputFormat(node, output_idx));
  kernel_info->SetOutputKernelTensor(kernel_tensor, output_idx);

  return kernel_info->GetOutputKernelTensor(output_idx);
}

const KernelTensorPtr &AnfRuntimeAlgorithm::GetPrevNodeOutputKernelTensor(const AnfNodePtr &node, size_t input_idx,
                                                                          bool skip_nop_node) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx, false);
  return GetOutputKernelTensor(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
}

const KernelTensorPtr &AnfRuntimeAlgorithm::GetOrCreatePrevNodeOutputKernelTensor(const AnfNodePtr &node,
                                                                                  size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx, false);
  return GetOrCreateOutputKernelTensor(kernel_with_index.first, kernel_with_index.second);
}

std::vector<KernelTensor *> AnfRuntimeAlgorithm::GetOrCreateAllInputKernelTensors(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  std::vector<KernelTensor *> input_kernel_tensors(input_num);
  for (size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    input_kernel_tensors[input_idx] = GetOrCreatePrevNodeOutputKernelTensor(node, input_idx).get();
  }
  return input_kernel_tensors;
}

std::vector<KernelTensor *> AnfRuntimeAlgorithm::GetOrCreateAllOutputKernelTensors(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  std::vector<KernelTensor *> output_kernel_tensors(output_num);
  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    output_kernel_tensors[output_idx] = GetOrCreateOutputKernelTensor(node, output_idx).get();
  }
  return output_kernel_tensors;
}

KernelTensorPtr AnfRuntimeAlgorithm::CreateOutputKernelTensorWithDeviceInfo(
  const AnfWithOutIndex &node_with_index, void *const device_ptr, size_t size, const string &format, TypeId dtype_id,
  const ShapeVector &host_shape, const std::string &device_name, uint32_t device_id, const UserDataPtr &user_data,
  uint32_t stream_id) {
  abstract::BaseShapePtr shape;
  TypePtr type;
  ValuePtr value;
  TensorStorageInfoPtr info = nullptr;
  if (AnfAlgo::ExistOutputKernelTensor(node_with_index.first, node_with_index.second)) {
    const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node_with_index.first, node_with_index.second, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor->GetShape());
    MS_EXCEPTION_IF_NULL(kernel_tensor->GetType());
    shape = kernel_tensor->GetShape()->Clone();
    type = kernel_tensor->GetType()->Clone();
    value = kernel_tensor->GetValueTrack();
    info = kernel_tensor->tensor_storage_info();
  }
  if (value == nullptr) {
    std::tie(shape, type, value) = AnfAlgo::GetAbstractInfo(node_with_index.first, node_with_index.second);
  }

  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(type);
  MS_LOG(DEBUG) << "Create device address for node: " << node_with_index.first->fullname_with_scope()
                << ", output index: " << node_with_index.second << ", device ptr: " << device_ptr << ", size: " << size
                << ", host shape: " << host_shape << ", format: " << format << ", dtype id: " << dtype_id
                << ", device name: " << device_name << ", device id: " << device_id << ", stream id: " << stream_id
                << ", Shape: " << shape->ToString() << ", Type: " << type->ToString()
                << ", Value: " << (value ? value->ToString() : "nullptr");

  auto out_tensor = CreateKernelTensor(shape, type, value, device_ptr, size, format, dtype_id, host_shape, device_name,
                                       device_id, user_data);
  if (info) {
    out_tensor->set_tensor_storage_info(info);
  }
  return out_tensor;
}

KernelTensorPtr AnfRuntimeAlgorithm::CreateKernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type,
                                                        const ValuePtr &value, void *device_ptr, size_t size,
                                                        const std::string &format, TypeId dtype_id,
                                                        const ShapeVector &host_shape, const string &device_name,
                                                        uint32_t device_id, const UserDataPtr &user_data) {
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto device_address =
    res_manager->CreateDeviceAddress(device_ptr, size, host_shape, kernel::GetFormatFromStrToEnum(format), dtype_id,
                                     device_name, device_id, 0, user_data);
  // Currently, address_common and device_address are not unified. Kernel tensor may use info from address_common
  // or device_address, so all info keep to kernel tensor.
  // Only device address are keep for construct after unified.
  auto kernel_tensor = std::make_shared<kernel::KernelTensor>(device_address, shape, type, value, device_ptr, size,
                                                              format, dtype_id, host_shape, device_name, device_id);
  return kernel_tensor;
}

KernelTensorPtr AnfRuntimeAlgorithm::CreateKernelTensor(void *device_ptr, size_t size, Format format, TypeId dtype_id,
                                                        const ShapeVector &host_shape, const string &device_name,
                                                        uint32_t device_id, const UserDataPtr &user_data) {
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto device_address = res_manager->CreateDeviceAddress(device_ptr, size, host_shape, format, dtype_id, device_name,
                                                         device_id, 0, user_data);
  auto kernel_tensor = std::make_shared<kernel::KernelTensor>(device_address, dtype_id, host_shape);
  return kernel_tensor;
}

std::vector<size_t> AnfRuntimeAlgorithm::GetNodeInputSizeList(const AnfNodePtr &node) {
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(node);
  size_t input_num = input_kernel_tensors.size();
  std::vector<size_t> input_size_list(input_num, 0);
  for (size_t i = 0; i < input_num; i++) {
    MS_EXCEPTION_IF_NULL(input_kernel_tensors[i]);
    input_size_list[i] = input_kernel_tensors[i]->size();
  }

  return input_size_list;
}

size_t AnfRuntimeAlgorithm::GetOutputAddressNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetOutputNum();
}

// set output device addr of anf_node
void AnfRuntimeAlgorithm::SetOutputAddr(const DeviceAddressPtr &addr, size_t output_idx, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->OutputKernelTensorExist(output_idx)) {
    MS_LOG(DEBUG) << "There is no kernel tensor for node: " << node->DebugString() << ", with index: " << output_idx
                  << ", new a kernel tensor.";
    abstract::BaseShapePtr shape;
    TypePtr type;
    ValuePtr value;
    std::tie(shape, type, value) = GetAbstractInfo(node, output_idx);
    MS_LOG(DEBUG) << "Create kernel tensor for node: " << node->fullname_with_scope()
                  << ", output index: " << output_idx << ", device address: " << addr.get()
                  << ", Shape: " << shape->ToString() << ", Type: " << type->ToString()
                  << ", Value: " << (value ? value->ToString() : "nullptr");

    MS_EXCEPTION_IF_NULL(shape);
    MS_EXCEPTION_IF_NULL(type);
    auto out_tensor = std::make_shared<kernel::KernelTensor>(shape, type, value);
    out_tensor->set_device_address(addr);
    SetOutputKernelTensor(out_tensor, output_idx, node.get());
  }
  if (!kernel_info->SetOutputAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// set output kernel tensor of anf node
void AnfRuntimeAlgorithm::SetOutputKernelTensor(const KernelTensorPtr &kernel_tensor, size_t output_idx,
                                                AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetOutputKernelTensor(kernel_tensor, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// set workspace device addr of anf_node
void AnfRuntimeAlgorithm::SetWorkspaceAddr(const DeviceAddressPtr &addr, size_t output_idx, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->WorkspaceKernelTensorExist(output_idx)) {
    auto out_tensor = std::make_shared<kernel::KernelTensor>(addr);
    MS_LOG(DEBUG) << "Create kernel tensor: " << out_tensor << ", device address: " << addr;
    if (!kernel_info->SetWorkspaceKernelTensor(out_tensor, output_idx)) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                        << trace::DumpSourceLines(node);
    }
  }
  if (!kernel_info->SetWorkspaceAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// set workspace kernel tensor of anf node
void AnfRuntimeAlgorithm::SetWorkspaceKernelTensor(const KernelTensorPtr &kernel_tensor, size_t output_idx,
                                                   AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetWorkspaceKernelTensor(kernel_tensor, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// get workspace device addr of anf_node
DeviceAddress *AnfRuntimeAlgorithm::GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetWorkspaceAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output_idx " << output_idx << " of node " << node->DebugString()
                                      << "] workspace addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

// get workspace kernel tensor of anf_node
KernelTensorPtr AnfRuntimeAlgorithm::GetWorkspaceKernelTensor(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &kernel_tenosr = kernel_info->GetWorkspaceKernelTensor(output_idx);
  if (kernel_tenosr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Output_idx " << output_idx << " of node " << node->DebugString()
                                      << "] workspace addr is not exist." << trace::DumpSourceLines(node);
  }
  return kernel_tenosr;
}

// get workspace device mutable addr of anf_node
DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableWorkspaceAddr(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableWorkspaceAddr(index);
  if (addr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Index " << index << " of node " << node->DebugString()
                                      << "] workspace addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

kernel::OpPattern AnfRuntimeAlgorithm::GetOpPattern(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->op_pattern();
}

// get KernelBuildType of node, such as ATT,RT,FWK and so on
KernelType AnfRuntimeAlgorithm::GetKernelType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG(DEBUG) << "Node: " << node->fullname_with_scope() << " has no kernel build info, using UNKNOWN_KERNEL_TYPE";
    return KernelType::UNKNOWN_KERNEL_TYPE;
  }
  return build_info->kernel_type();
}

void AnfRuntimeAlgorithm::SetFusionType(const AnfNodePtr &node, const std::string &type) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetFusionType(type);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

void AnfRuntimeAlgorithm::SetCoreType(const AnfNodePtr &node, const std::string &core_type) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetCoreType(core_type);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

std::string AnfRuntimeAlgorithm::GetCoreType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return "";
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->core_type();
}

kernel::OpType AnfRuntimeAlgorithm::GetOpType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->op_type();
}

void AnfRuntimeAlgorithm::SetOutputDataDesc(const AnfNodePtr &node, const std::vector<nlohmann::json> &desc) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetOutputDataDesc(desc);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

std::vector<nlohmann::json> AnfRuntimeAlgorithm::GetOutputDataDesc(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return {};
  }
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return {};
  }
  return build_info->output_data_desc();
}

kernel::Processor AnfRuntimeAlgorithm::GetProcessor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->processor();
}

std::string AnfRuntimeAlgorithm::GetFusionType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return kPatternUnknown;
  }
  return build_info->fusion_type();
}

// set select kernel_build_info
void AnfRuntimeAlgorithm::SetSelectKernelBuildInfo(const KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (kernel_info->has_build_info() && (select_kernel_build_info != nullptr)) {
    auto ori_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
    auto input_object_types = ori_kernel_build_info->GetAllInputKernelObjectTypes();
    auto output_object_types = ori_kernel_build_info->GetAllOutputKernelObjectTypes();
    if (!input_object_types.empty() && select_kernel_build_info->GetAllInputKernelObjectTypes().empty()) {
      select_kernel_build_info->SetInputsKernelObjectType(input_object_types);
    }
    if (!output_object_types.empty() && select_kernel_build_info->GetAllOutputKernelObjectTypes().empty()) {
      MS_LOG(DEBUG) << "set kernel object type:" << output_object_types << " for node:" << node->fullname_with_scope();
      select_kernel_build_info->SetOutputsKernelObjectType(output_object_types);
    }
  }
  return kernel_info->set_select_kernel_build_info(select_kernel_build_info);
}

// get select kernel_build_info
KernelBuildInfoPtr AnfRuntimeAlgorithm::GetSelectKernelBuildInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->GetMutableSelectKernelBuildInfo();
}

// get kernelMode
KernelMod *AnfRuntimeAlgorithm::GetKernelMod(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->MutableKernelMod();
}

// set kernel mod
void AnfRuntimeAlgorithm::SetKernelMod(const KernelModPtr &kernel_mod, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_kernel_mod(kernel_mod);
}

void AnfRuntimeAlgorithm::SetStreamId(uint32_t stream_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_id(stream_id);
}

uint32_t AnfRuntimeAlgorithm::GetStreamId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_id();
}

void AnfRuntimeAlgorithm::SetStreamDistinctionLabel(uint32_t stream_label, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_distinction_label(stream_label);
}

uint32_t AnfRuntimeAlgorithm::GetStreamDistinctionLabel(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_distinction_label();
}

void AnfRuntimeAlgorithm::SetGraphId(uint32_t graph_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_graph_id(graph_id);
}

uint32_t AnfRuntimeAlgorithm::GetGraphId(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->graph_id();
}

bool AnfRuntimeAlgorithm::IsFeatureMapOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    ValuePtr value = value_node->value();
    std::vector<tensor::TensorPtr> tensors;
    TensorValueToTensor(value, &tensors);
    auto ret = false;
    if (!tensors.empty()) {
      auto all_tensor_have_address = true;
      for (const auto &tensor : tensors) {
        MS_EXCEPTION_IF_NULL(tensor);
        if (tensor->device_address() == nullptr) {
          all_tensor_have_address = false;
          break;
        }
      }
      ret = all_tensor_have_address;
    }
    return ret;
  }
  if (IsPrimitiveCNode(node, prim::kPrimLoad) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return IsFeatureMapOutput(node->cast<CNodePtr>()->input(1));
  }
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  // If node is a call node which not have kernel info
  if (kernel_info == nullptr) {
    return false;
  }
  return kernel_info->is_feature_map();
}

bool AnfRuntimeAlgorithm::IsFeatureMapInput(const AnfNodePtr &node, size_t input_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node)
      << "Cannot input a parameter or a valuenode to charge it's input if is a feature map."
      << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(input_index + 1);
  return IsFeatureMapOutput(input_node);
}

size_t AnfRuntimeAlgorithm::GetInputGraphIdxByKernelIdx(const mindspore::AnfNodePtr &anf_node,
                                                        size_t input_index_in_kernel) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return input_index_in_kernel;
}

size_t AnfRuntimeAlgorithm::GetInputKernelIdxByGraphIdx(const mindspore::AnfNodePtr &anf_node,
                                                        size_t input_index_in_graph) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return input_index_in_graph;
}

std::vector<KernelGraphPtr> AnfRuntimeAlgorithm::GetCallSwitchKernelGraph(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!(common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
        common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer))) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Node: " << cnode->DebugString()
                                       << "is not a call or switch or switch_layer node."
                                       << trace::DumpSourceLines(cnode);
  }
  auto get_switch_kernel_graph = [cnode](size_t input_index) -> KernelGraphPtr {
    auto partial = cnode->input(input_index);
    MS_EXCEPTION_IF_NULL(partial);
    if (IsValueNode<KernelGraph>(partial)) {
      return GetValueNode<KernelGraphPtr>(partial);
    }
    auto partial_cnode = partial->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_cnode);
    auto graph_node = partial_cnode->input(kPartialGraphIndex);
    MS_EXCEPTION_IF_NULL(graph_node);
    auto graph_value_node = graph_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(graph_value_node);
    auto graph_value = graph_value_node->value();
    MS_EXCEPTION_IF_NULL(graph_value);
    auto child_graph = graph_value->cast<KernelGraphPtr>();
    return child_graph;
  };
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    auto input1 = cnode->input(kPartialGraphIndex);
    MS_EXCEPTION_IF_NULL(input1);
    auto value_node = input1->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto kernel_graph = value_node->value();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    return {kernel_graph->cast<KernelGraphPtr>()};
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    return {get_switch_kernel_graph(kSwitchTrueBranchIndex), get_switch_kernel_graph(kSwitchFalseBranchIndex)};
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer)) {
    std::vector<KernelGraphPtr> child_graphs;
    for (size_t idx = kSwitchLayerBranchesIndex; idx < cnode->size(); idx++) {
      auto child_graph = get_switch_kernel_graph(idx);
      child_graphs.emplace_back(child_graph);
    }
    return child_graphs;
  }
  return {};
}

KernelGraphPtr AnfRuntimeAlgorithm::GetValueNodeKernelGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  auto kernel_graph = value->cast<KernelGraphPtr>();
  return kernel_graph;
}

bool AnfRuntimeAlgorithm::IsIndependentNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetKernelType(node) != AICPU_KERNEL) {
    return false;
  }

  if (common::AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
    MS_LOG(INFO) << "GetNext should not be independent node";
    return false;
  }

  // aicpu stack ops are not independent nodes.
  if (common::AnfAlgo::GetCNodeName(node) == kStackInitOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackDestroyOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackPopOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackPushOpName) {
    MS_LOG(INFO) << "AICPU stack ops should not be independent node";
    return false;
  }

  size_t input_nums = common::AnfAlgo::GetInputTensorNum(node);
  if (input_nums == 0) {
    return true;
  }

  auto inputs = node->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      return false;
    }
  }
  return true;
}

KernelGraphPtr AnfRuntimeAlgorithm::FetchKernelGraph(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &func_graph = node->func_graph();
  if (func_graph == nullptr) {
    return nullptr;
  } else {
    return func_graph->cast<KernelGraphPtr>();
  }
}

AnfNodePtr AnfRuntimeAlgorithm::FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  auto front_node_with_index = graph.GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first != nullptr) {
    return front_node_with_index.first;
  }

  auto front_node = graph.GetFrontAnfByBackendAnf(backend_node);
  // PyNative forward graph does not has front node, using backend node instead.
  if (front_node == nullptr) {
    front_node = backend_node;
  }
  return front_node;
}

namespace {
// Host kernel with inputs on host
bool SkipDataSync(const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &depend_tensors) {
  if (!common::AnfAlgo::IsHostKernel(node)) {
    return false;
  }
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto real_input = input_with_index.first;
    auto iter_tensor = depend_tensors.find(i);
    if (iter_tensor != depend_tensors.end()) {
      auto output_addr = AnfAlgo::GetOutputAddr(real_input, 0);
      MS_EXCEPTION_IF_NULL(output_addr);
      if (output_addr->GetDeviceType() != device::DeviceType::kCPU) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

void AnfRuntimeAlgorithm::InferShape(const CNodePtr &node, std::map<uint32_t, tensor::TensorPtr> *depend_tensors) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "InferShape start, node:" << node->DebugString();
  auto inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Inputs should not be empty! Cnode: " << node->DebugString() << "."
                                      << trace::DumpSourceLines(node);
  }
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto real_input = input_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    auto cnode_input = node->input(i + 1);
    MS_EXCEPTION_IF_NULL(cnode_input);
    if (depend_tensors != nullptr) {
      auto iter_tensor = depend_tensors->find(i);
      if (iter_tensor != depend_tensors->cend()) {
        auto tensor_ptr = iter_tensor->second;
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        if (!SkipDataSync(node, *depend_tensors)) {
          // sync data from device to host
          tensor_ptr->data_sync();
        }
        // cppcheck-suppress unreadVariable
        auto lock = AnfUtils::GetAbstractLock(real_input.get());
        auto real_abs = real_input->abstract();
        if (real_abs->isa<abstract::AbstractTensor>()) {
          real_abs->set_value(tensor_ptr);
        } else if (real_abs->isa<abstract::AbstractTuple>() && (!common::AnfAlgo::IsDynamicSequence(real_input))) {
          auto tuple_get_item_index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode_input->cast<CNodePtr>());
          auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
          MS_EXCEPTION_IF_NULL(abstract_tuple);
          auto tuple_elements = abstract_tuple->elements()[tuple_get_item_index];
          tuple_elements->set_value(tensor_ptr);
        }
      }
    }
    common::AnfAlgo::AddArgList(&args_spec_list, real_input, input_with_index.second);
  }
  auto eval_result = opt::CppInferShapeAndType(primitive, args_spec_list);
  node->set_abstract(eval_result);
}

void AnfRuntimeAlgorithm::InsertMakeTupleForOutput(const NotNull<KernelGraphPtr> &root_graph) {
  auto return_node = root_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() <= kReturnDataIndex) {
    return;
  }
  auto make_tuple = root_graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), root_graph->output()});
  MS_EXCEPTION_IF_NULL(root_graph->output());
  MS_EXCEPTION_IF_NULL(make_tuple);
  abstract::AbstractBasePtrList abs_list{root_graph->output()->abstract()};
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  root_graph->set_output(make_tuple);
}

std::string AnfAlgo::GetJitLevel(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->cast<KernelGraphPtr>()) {
    const auto &backend_jit_config = graph->cast<KernelGraphPtr>()->backend_jit_config();
    if (!backend_jit_config.jit_level.empty()) {
      return backend_jit_config.jit_level;
    }
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string jit_level = context->GetJitLevel();
  return jit_level;
}

std::string AnfAlgo::GetBackend(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->cast<KernelGraphPtr>()) {
    const auto &backend_jit_config = graph->cast<KernelGraphPtr>()->backend_jit_config();
    if (!backend_jit_config.backend.empty()) {
      return backend_jit_config.backend;
    }
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->GetBackend();
  return backend;
}

bool AnfAlgo::GetDisableFormatTransform(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &backend_jit_config = graph->backend_jit_config();
  auto disable_format_transform = backend_jit_config.disable_format_transform;
  if (disable_format_transform == false) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    disable_format_transform = context->get_param<bool>(MS_CTX_DISABLE_FORMAT_TRANSFORM);
  }
  return disable_format_transform;
}

std::string AnfAlgo::GetExecOrderAlgo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &backend_jit_config = graph->backend_jit_config();
  auto exec_order = backend_jit_config.exec_order;
  if (exec_order.empty()) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    exec_order = context->get_param<std::string>(MS_CTX_EXEC_ORDER);
  }
  return exec_order;
}

std::map<std::string, std::map<std::string, std::string>> AnfAlgo::GetGeOptions(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &backend_jit_config = graph->backend_jit_config();
  return backend_jit_config.ge_options;
}

std::map<std::string, std::string> AnfAlgo::GetGeOptions(std::string option_level) {
  std::map<std::string, std::string> ret;
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  if (backend_jit_config.ge_options.find(option_level) != backend_jit_config.ge_options.end()) {
    ret = backend_jit_config.ge_options.at(option_level);
  }

  if (ret.empty()) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    const auto &ge_options_str = context->get_param<std::string>(MS_CTX_GE_OPTIONS);
    if (!ge_options_str.empty()) {
      nlohmann::json options_json = nlohmann::json::parse(ge_options_str);
      if (options_json.contains(option_level)) {
        options_json[option_level].get_to(ret);
      }
    }
  }
  return ret;
}

void AnfRuntimeAlgorithm::UpdateGraphValidRefPair(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  const auto &origin_ref_map = graph->GetRefMap();
  std::map<AnfWithOutIndex, AnfWithOutIndex> new_ref_map;
  for (const auto &node : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(node);
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
    if (output_num == 0) {
      MS_LOG(DEBUG) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_num; ++i) {
      session::AnfWithOutIndex out_pair(node, i);
      auto iter = origin_ref_map.find(out_pair);
      if (iter != origin_ref_map.end()) {
        auto ret = new_ref_map.try_emplace(iter->first, iter->second);
        if (!ret.second) {
          MS_LOG(WARNING) << "Duplicate ref_map key, node:" << node->fullname_with_scope() << " index:" << i;
        }
      }
    }
  }
  graph->set_ref_out_in_map(new_ref_map);
}

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(bool skip_mode, const ShapeVector &axes_shape) {
  // Skip run ReduceSum when axis is a Empty Tensor
  if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; }) && skip_mode) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if ((op_name != kReduceSumOpName) && (op_name != kReduceSumDOpName)) {
    return false;
  }

  bool skip_mode = false;
  if (common::AnfAlgo::HasNodeAttr(kAttrSkipMode, cnode)) {
    skip_mode = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrSkipMode);
  }

  if (!skip_mode) {
    return false;
  }

  const size_t axes_index = 1;
  if (cnode->size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(input_axes.get());
  auto abs = input_axes->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto axes_abs = abs->Clone();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; })) {
      return true;
    }
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsNeedUpdateShapeAndTypeAfterLaunch(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto graph = FetchKernelGraph(node.get());
  // The graph run mode does not have kernelmod.
  if ((graph == nullptr) || graph->is_graph_run_mode()) {
    return true;
  }

  auto kernel_mod = GetKernelMod(node);
  if (kernel_mod == nullptr) {
    return true;
  }
  return kernel_mod->IsNeedUpdateOutputShapeAndSize();
}

bool AnfRuntimeAlgorithm::HasComputedDependInputNode(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto real_input_num = common::AnfAlgo::GetInputTensorNum(kernel);

  for (size_t i = 0; i < real_input_num; i++) {
    const auto &input_node = common::AnfAlgo::GetInputNode(kernel, i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(real_input_node.first);
    if (!real_input_node.first->isa<CNode>()) {
      continue;
    }

    auto kernel_mod = AnfAlgo::GetKernelMod(real_input_node.first);
    if (kernel_mod && kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
      return true;
    }
  }
  return false;
}

void AnfRuntimeAlgorithm::UpdateOutputAddrSize(device::KernelInfo const *kernel_info, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  MS_EXCEPTION_IF_NULL(kernel);
  auto &output_kernel_tensors = kernel_info->output_kernel_tensor_list();
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_address = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_address);
    auto output_addr_size = AnfAlgo::GetOutputTensorMemSize(kernel, i);
    MS_LOG(DEBUG) << "output size:" << output_addr_size << " index:" << i
                  << " for kernel:" << kernel->fullname_with_scope()
                  << " abstract:" << (kernel->abstract() == nullptr ? "null" : kernel->abstract()->ToString());
    if (output_addr_size != output_address->GetSize()) {
      output_address->SetSize(output_addr_size);
    }
  }
}

void AnfRuntimeAlgorithm::AddOutInRefToGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto kernel_info = dynamic_cast<device::KernelInfo *>(cnode->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    for (const auto &ref : kernel_info->out_in_ref_map()) {
      size_t output_index = ref.first;
      size_t input_index = ref.second;
      auto final_pair = std::make_pair(cnode, output_index);
      auto origin_pair = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0);
      MS_LOG(INFO) << "The reference relation output " << final_pair.first->fullname_with_scope()
                   << ", output index: " << final_pair.second << " to input "
                   << origin_pair.first->fullname_with_scope() << ", output index: " << origin_pair.second;
      // Add to graph only if the input is not a monad.
      if (!HasAbstractUMonad(origin_pair.first) && !HasAbstractIOMonad(origin_pair.first)) {
        graph->AddRefCorrespondPairs(final_pair, origin_pair);
      }
    }
  }
}

bool AnfRuntimeAlgorithm::HasOriginFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return anf_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrOriginFormat, anf_node->cast<CNodePtr>());
}

std::string AnfRuntimeAlgorithm::GetOriginFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrOriginFormat, anf_node->cast<CNodePtr>())) {
    return common::AnfAlgo::GetNodeAttr<std::string>(anf_node, kAttrOriginFormat);
  }
  return {};
}

bool AnfRuntimeAlgorithm::NodeValueIsFuncGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value().get();
  MS_EXCEPTION_IF_NULL(value);
  return value->isa<FuncGraph>();
}

bool AnfRuntimeAlgorithm::IsNodeSupportKernelSelectBackoff(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  static std::string disable_kernel_backoff;
  static bool first_get_backoff_env = true;
  if (first_get_backoff_env) {
    disable_kernel_backoff = common::GetEnv(kDisableKernelBackoff);
    first_get_backoff_env = false;
  }
  if (disable_kernel_backoff == "1" && (!common::AnfAlgo::IsTypeTransformOp(common::AnfAlgo::GetCNodeName(node)))) {
    MS_LOG(INFO) << "MS_DISABLE_KERNEL_BACKOFF has been set to turn off the kernel backoff ability.";
    return false;
  }

  if (graph == nullptr) {
    return false;
  }
  if (graph->is_from_single_op() || graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    MS_LOG(INFO) << "The pynative single op does not support the kernel backoff ability for graph:"
                 << graph->graph_id();
    return false;
  }
  return true;
}

void AnfRuntimeAlgorithm::SetKernelSelectBackoffInfo(const CNodePtr &node,
                                                     const std::pair<std::string, ExceptionType> &failure_info) {
  MS_EXCEPTION_IF_NULL(node);
  common::AnfAlgo::SetNodeAttr(kAttrKernelBackoffWithFailureInfo, MakeValue(failure_info.first), node);
  common::AnfAlgo::SetNodeAttr(kAttrKernelBackoffWithFailureType, MakeValue(static_cast<int32_t>(failure_info.second)),
                               node);
}

std::pair<std::string, ExceptionType> AnfRuntimeAlgorithm::GetKernelSelectBackoffInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsKernelSelectBackoffOp(node)) {
    return {"", NoExceptionType};
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto failure_info = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrKernelBackoffWithFailureInfo);
  auto failure_type =
    static_cast<ExceptionType>(common::AnfAlgo::GetNodeAttr<int32_t>(node, kAttrKernelBackoffWithFailureType));
  return {failure_info, failure_type};
}

bool AnfRuntimeAlgorithm::IsKernelSelectBackoffOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrKernelBackoffWithFailureInfo, cnode) &&
      common::AnfAlgo::HasNodeAttr(kAttrKernelBackoffWithFailureType, cnode)) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsNeedContinuesMemoryOp(const AnfNodePtr &kernel) {
  static std::set<std::string> names = {kMatMulAllReduceOpName, kAlltoAllVOpName, kAllGatherVOpName,
                                        kReduceScatterVOpName};
  bool flag = (common::AnfAlgo::IsCommunicationOp(kernel)) && (names.count(common::AnfAlgo::GetCNodeName(kernel)) == 0);
  return flag;
}

std::string AnfRuntimeAlgorithm::FetchDeviceTarget(const AnfNodePtr &node, const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  // The parameter also may be have the user data to express device target.
  auto ud_target = node->user_data<std::string>(kAttrPrimitiveTarget);
  if (ud_target != nullptr) {
    return *ud_target;
  }

  if (!node->isa<CNode>()) {
    return device::GetDeviceNameByType(graph->device_target());
  }

  // Only the CPU support kernel backoff.
  if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
    return kCPUDevice;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, cnode)) {
    return common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrPrimitiveTarget);
  }

  return device::GetDeviceNameByType(graph->device_target());
}

void AnfRuntimeAlgorithm::SetParameterDeviceTarget(const KernelGraphPtr graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = MakeManager({graph});
    graph->set_manager(manager);
  }

  const auto &graph_device_target = device::GetDeviceNameByType(graph->device_target());
  for (auto &input_node : graph->input_nodes()) {
    const auto &iter = manager->node_users().find(input_node);
    if (iter == manager->node_users().end()) {
      continue;
    }

    std::string device_target_affinity = graph_device_target;
    for (const auto &user_node : iter->second) {
      if (!AnfUtils::IsRealCNodeKernel(user_node.first)) {
        continue;
      }
      device_target_affinity = FetchDeviceTarget(user_node.first, graph.get());
      // If there is node with the same device target as the graph, then select the device target of graph affinity.
      if (device_target_affinity == graph_device_target) {
        break;
      }
    }

    // Set the device target for parameter when it is different with the graph.
    if (device_target_affinity != graph_device_target) {
      MS_LOG(INFO) << "Set the affinity device target for parameter:" << input_node->fullname_with_scope()
                   << " in graph:" << graph->graph_id() << " from graph device target:" << graph_device_target
                   << " to real device target:" << device_target_affinity;
      input_node->set_user_data(kAttrPrimitiveTarget, std::make_shared<std::string>(device_target_affinity));
    }
  }
}

TypeId AnfRuntimeAlgorithm::GetAbstractObjectType(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return kTypeUnknown;
  }
  if (abstract->isa<AbstractTensor>()) {
    return kObjectTypeTensorType;
  }
  if (abstract->isa<AbstractTuple>()) {
    return kObjectTypeTuple;
  }
  if (abstract->isa<abstract::AbstractList>()) {
    return kObjectTypeList;
  }
  if (abstract->isa<abstract::AbstractScalar>()) {
    // scalar input may not converted to tensor
    return kObjectTypeNumber;
  }
  if (abstract->isa<abstract::AbstractNone>()) {
    return kMetaTypeNone;
  }

  return kTypeUnknown;
}

TypeId AnfRuntimeAlgorithm::GetOutputObjectType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  if (abstract->isa<AbstractTuple>()) {
    auto tuple_abs = abstract->cast<abstract::AbstractTuplePtr>();
    auto items = tuple_abs->elements();
    MS_EXCEPTION_IF_CHECK_FAIL(output_idx < items.size(), "invalid output_idx");
    return AnfAlgo::GetAbstractObjectType(items[output_idx]);
  }
  if (output_idx != 0) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << node->DebugString() << "invalid output_idx" << trace::DumpSourceLines(node);
  }
  return AnfAlgo::GetAbstractObjectType(abstract);
}

TypeId AnfRuntimeAlgorithm::GetInputObjectType(const CNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_node = common::AnfAlgo::GetInputNode(node, input_idx);
  const std::vector<PrimitivePtr> need_handled_prims = {prim::kPrimMakeTuple, prim::kPrimTupleGetItem};
  auto real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false, need_handled_prims).first;
  return AnfAlgo::GetAbstractObjectType(real_input_node->abstract());
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllInputObjectType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << node->DebugString() << "anf_node is not CNode."
                                      << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  std::vector<TypeId> obj_types;
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    obj_types.push_back(AnfAlgo::GetInputObjectType(cnode, index));
  }
  return obj_types;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputObjectType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetOutputElementNum(node) == 0 && node->abstract() != nullptr &&
      !node->abstract()->isa<abstract::AbstractSequence>()) {
    return {};
  }
  return {AnfAlgo::GetAbstractObjectType(node->abstract())};
}

abstract::BaseShapePtr AnfRuntimeAlgorithm::GetOutputDetailShape(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    if (output_idx == 0) {
      return base_shape;
    }
    MS_LOG_WITH_NODE(EXCEPTION, node) << "The node " << node->DebugString() << "is a single output node but got index ["
                                      << output_idx << "]." << trace::DumpSourceLines(node);
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (IsRealSquenceOutput(node)) {
      return tuple_shape;
    }
    if (output_idx >= tuple_shape->size()) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Output index " << output_idx << "is larger than output number "
                                        << tuple_shape->size() << " node:" << node->DebugString() << "."
                                        << trace::DumpSourceLines(node);
    }
    auto b_shp = (*tuple_shape)[output_idx];
    if (b_shp->isa<abstract::Shape>() || b_shp->isa<abstract::NoShape>() || b_shp->isa<abstract::TupleShape>() ||
        b_shp->isa<abstract::DynamicSequenceShape>()) {
      return b_shp;
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "The output type of node index:" << output_idx
                                        << " should be a NoShape , ArrayShape or a TupleShape, but it is "
                                        << base_shape->ToString() << "node :" << node->DebugString() << "."
                                        << trace::DumpSourceLines(node);
    }
  } else if (base_shape->isa<abstract::NoShape>()) {
    return base_shape;
  } else if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    return common::AnfAlgo::GetDynamicSequenceShape(node, output_idx);
  }
  MS_LOG_WITH_NODE(EXCEPTION, node)
    << "The output type of node should be a NoShape , ArrayShape or a TupleShape, but it is " << base_shape->ToString()
    << " node : " << node->DebugString() << trace::DumpSourceLines(node);
}

abstract::BaseShapePtr AnfRuntimeAlgorithm::GetPrevNodeOutputDetailShape(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfAlgo::GetOutputDetailShape(kernel_with_index.first, kernel_with_index.second);
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputInferDataTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypeId> outputs;
  auto out_nums = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < out_nums; i++) {
    auto type = common::AnfAlgo::GetOutputInferDataType(node, i);
    outputs.push_back(type);
  }
  return outputs;
}

// if input node is MakeTuple, find the PrevNodeNum recursively;
// The monad node in the end is not included in the num;
size_t AnfRuntimeAlgorithm::GetInputElementNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t element_num = 0;
  size_t input_num = cnode->size() - 1;
  bool cal_monad_flag = false;
  for (size_t i = input_num; i > 0; --i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, i - 1);
    if (!cal_monad_flag && HasAbstractMonad(input_node)) {
      continue;
    } else if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
      element_num += GetInputElementNum(input_node);
      cal_monad_flag = true;
    } else if (common::AnfAlgo::IsTupleOutput(input_node)) {
      element_num += AnfAlgo::GetOutputElementNum(input_node);
      cal_monad_flag = true;
    } else {
      ++element_num;
      cal_monad_flag = true;
    }
  }

  return element_num;
}

void AnfRuntimeAlgorithm::SetDynamicAttrToPrim(const PrimitivePtr &prim) {
  (void)prim->AddAttr(kAttrMutableKernel, MakeValue(true));
  (void)prim->AddAttr(kAttrInputIsDynamicShape, MakeValue(true));
  (void)prim->AddAttr(kAttrOutputIsDynamicShape, MakeValue(true));
}

bool AnfRuntimeAlgorithm::IsScalarConvertToTensor(const AnfNodePtr &input_node, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(node);
  if (!input_node->isa<ValueNode>()) {
    return false;
  }

  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<Scalar>()) {
    return false;
  }

  const auto &abs = node->abstract();
  if (ContainScalarOut(abs)) {
    MS_LOG(INFO) << "The input scalar value node:" << input_node->fullname_with_scope()
                 << " of cnode:" << node->fullname_with_scope() << " doesn't need convert to tensor.";
    return false;
  }
  return true;
}

bool AnfRuntimeAlgorithm::IsSequenceOutputOfScalar(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abs = node->abstract();
  if (abs == nullptr || !abs->isa<abstract::AbstractSequence>()) {
    return false;
  }
  // Check all elements in tuple/list are scalar.
  auto abs_seq = abs->cast_ptr<abstract::AbstractSequence>();
  MS_EXCEPTION_IF_NULL(abs_seq);
  if (abs_seq->dynamic_len()) {
    const auto &element_abs = abs_seq->dynamic_len_element_abs();
    return (element_abs == nullptr) || (element_abs->isa<abstract::AbstractScalar>());
  }
  const auto &elements = abs_seq->elements();

  return std::all_of(elements.begin(), elements.end(), [](const AbstractBasePtr &element) {
    return (element != nullptr) && (element->isa<abstract::AbstractScalar>()) &&
           (element->BuildValue() == nullptr || (!element->BuildValue()->isa<StringImm>()));
  });
}

bool AnfRuntimeAlgorithm::IsSummaryNode(const AnfNodePtr &node) {
  return (IsPrimitiveCNode(node, prim::kPrimScalarSummary) || IsPrimitiveCNode(node, prim::kPrimTensorSummary) ||
          IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary));
}

namespace {
bool CheckValidTensorTuple(const std::vector<ValuePtr> &values) {
  if (values.empty() || values[0] == nullptr || (!values[0]->isa<tensor::Tensor>())) {
    return false;
  }
  const auto &const_tensor = values[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(const_tensor);
  const auto &const_shape = const_tensor->shape();
  const auto &const_type_id = const_tensor->data_type();
  size_t const_size = const_tensor->Size();
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] == nullptr || (!values[i]->isa<tensor::Tensor>())) {
      MS_LOG(ERROR) << "Invalid value:" << (values[i] == nullptr ? "nullptr" : values[i]->ToString()) << " index:" << i
                    << " in value tuple";
      return false;
    }
    const auto &tensor = values[i]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &shape = tensor->shape();
    const auto &type_id = tensor->data_type();
    size_t size = tensor->Size();
    if (shape != const_shape || type_id != const_type_id || size != const_size) {
      return false;
    }
  }
  return true;
}

// Return a new tensor with type like single_value.
void SetScalarToTensor(const std::vector<ValuePtr> &values, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &tensor_type_id = tensor->data_type();
  const auto dst_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(dst_ptr);
  MS_LOG(DEBUG) << "Set scalar tuple to tensor, dst size:" << tensor->data().nbytes();
  for (size_t i = 0; i < values.size(); ++i) {
    // Check mem size.
    if (SizeToLong(abstract::TypeIdSize(tensor_type_id) * (i + 1)) > tensor->data().nbytes()) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Value size:" << values.size()
                                 << " type:" << tensor_type_id << " out of range:" << tensor->data().nbytes();
    }
    const auto &value = values[i];
    MS_EXCEPTION_IF_NULL(value);
    // Check value type.
    if (value->type()->type_id() != tensor_type_id) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid value type:" << value->type()->type_id()
                                 << " for value:" << value->ToString() << " dst type:" << tensor_type_id;
    }
    if (tensor_type_id == TypeId::kNumberTypeInt8) {
      (reinterpret_cast<int8_t *>(dst_ptr))[i] = GetValue<int8_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt16) {
      (reinterpret_cast<int16_t *>(dst_ptr))[i] = GetValue<int16_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt32 || tensor_type_id == kNumberTypeInt) {
      (reinterpret_cast<int32_t *>(dst_ptr))[i] = GetValue<int32_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt64) {
      (reinterpret_cast<int64_t *>(dst_ptr))[i] = GetValue<int64_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeBool) {
      (reinterpret_cast<bool *>(dst_ptr))[i] = GetValue<bool>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeFloat32 || tensor_type_id == TypeId::kNumberTypeFloat) {
      (reinterpret_cast<float *>(dst_ptr))[i] = GetValue<float>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeFloat64) {
      (reinterpret_cast<double *>(dst_ptr))[i] = GetValue<double>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt8) {
      (reinterpret_cast<uint8_t *>(dst_ptr))[i] = GetValue<uint8_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt16) {
      (reinterpret_cast<uint16_t *>(dst_ptr))[i] = GetValue<uint16_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt || tensor_type_id == TypeId::kNumberTypeUInt32) {
      (reinterpret_cast<uint32_t *>(dst_ptr))[i] = GetValue<uint32_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt64) {
      (reinterpret_cast<uint64_t *>(dst_ptr))[i] = GetValue<uint64_t>(value);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid tuple type:" << tensor_type_id
                                 << " for scalar to tensor.";
    }
  }
}
}  // namespace

tensor::TensorPtr AnfRuntimeAlgorithm::CreateMapTensor(const KernelTensorPtr &output_kernel_tensor) {
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  const auto &output_device_address = output_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(output_device_address);
  const auto &user_data = output_kernel_tensor->user_data();
  MS_EXCEPTION_IF_NULL(user_data);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
    auto shape_vector = user_data->get<ShapeVector>(kHashTableShapeVector);
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    auto default_value = user_data->get<Value>(kHashTableDefaultValue);
    MS_EXCEPTION_IF_NULL(shape_vector);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    MS_EXCEPTION_IF_NULL(default_value);
    auto map_tensor = std::make_shared<tensor::MapTensor>(*key_type, *value_type, *shape_vector, default_value);
    map_tensor->set_device_address(output_device_address);
    return map_tensor;
  }
  MS_LOG(WARNING) << "Invalid user data type:" << *user_data_type;
  return nullptr;
}

tensor::TensorPtr AnfRuntimeAlgorithm::CreateMapTensor(const AnfNodePtr &output_node, size_t output_index) {
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_index, false);
  return CreateMapTensor(kernel_tensor);
}

// In dynamic sequence, since the number of members is not determined in compile time, the entire sequence needs
// to be placed in single tensor, and the shape of the tuple needs to be recorded in the tensor, so that the shape
// of the tensor can be accurately restored during the dynamic shape derivation process in runtime.
tensor::TensorPtr AnfRuntimeAlgorithm::SequenceToTensor(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueSequence>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid sequence value:" << value->ToString();
  }

  const auto &sequence_value = value->cast<ValueSequencePtr>();
  const auto &values = sequence_value->value();
  if (values.empty()) {
    auto tensor = std::make_shared<tensor::Tensor>();
    abstract::BaseShapePtr base_shape = nullptr;
    if (value->isa<ValueTuple>()) {
      base_shape = std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList());
    } else {
      base_shape = std::make_shared<abstract::ListShape>(abstract::BaseShapePtrList());
    }
    tensor->set_base_shape(base_shape);
    return tensor;
  }
  if (values[0] == nullptr || ((!values[0]->isa<Scalar>()) && (!values[0]->isa<tensor::Tensor>()))) {
    MS_LOG(WARNING) << "Empty sequence in sequence value:" << value->ToString();
    return std::make_shared<tensor::Tensor>();
  }

  ShapeVector shape_vector{SizeToLong(values.size())};
  if (values[0]->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "Check dynamic tuple tensor";
    if (!CheckValidTensorTuple(values)) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid dynamic sequence tuple:"
                                 << value->ToString();
    }
    const auto &tensor = values[0]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t size = tensor->Size();
    const auto &type_id = tensor->data_type();
    auto single_shape_vector = tensor->shape();
    const auto &single_shape = std::make_shared<abstract::Shape>(single_shape_vector);
    (void)shape_vector.insert(shape_vector.end(), single_shape_vector.begin(), single_shape_vector.end());
    const auto &shape = std::make_shared<abstract::Shape>(shape_vector);
    auto new_tensor = std::make_shared<tensor::Tensor>(type_id, shape_vector);
    MS_EXCEPTION_IF_NULL(new_tensor);
    const auto dst_ptr = new_tensor->data_c();
    MS_EXCEPTION_IF_NULL(dst_ptr);
    MS_LOG(DEBUG) << "Copy start, dst size:" << new_tensor->data().nbytes();
    for (size_t i = 0; i < values.size(); ++i) {
      const auto &sub_value = values[i];
      MS_EXCEPTION_IF_NULL(sub_value);
      const auto &src_tensor = sub_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(src_tensor);
      MS_EXCEPTION_IF_NULL(src_tensor->data_c());
      auto ret = memcpy_s((reinterpret_cast<char *>(dst_ptr)) + i * size,
                          static_cast<size_t>(new_tensor->data().nbytes()), src_tensor->data_c(), size);
      if (ret != EOK) {
        MS_LOG(INTERNAL_EXCEPTION)
          << "#dmsg#Runtime error info:#dmsg#Failed to copy data into tensor, memcpy_s errorno: " << ret;
      }
    }
    const auto &element_shapes = std::vector<abstract::BaseShapePtr>(values.size(), single_shape);
    new_tensor->set_base_shape(std::make_shared<abstract::TupleShape>(element_shapes));
    MS_LOG(DEBUG) << "merge tensor from:" << value->ToString() << " to:" << new_tensor->ToString() << " tensor addr"
                  << new_tensor;
    return new_tensor;
  }

  // Create the tensor.
  auto tensor = std::make_shared<tensor::Tensor>(values[0]->type()->type_id(), shape_vector);
  MS_EXCEPTION_IF_NULL(tensor);
  SetScalarToTensor(values, tensor);
  // Build the tuple shape and set into tensor.
  const auto &element_shape = std::make_shared<abstract::Shape>(ShapeVector({}));
  const auto &element_shapes = std::vector<abstract::BaseShapePtr>(values.size(), element_shape);
  tensor->set_base_shape(std::make_shared<abstract::TupleShape>(element_shapes));
  return tensor;
}

void AnfRuntimeAlgorithm::FlattenDynamicInputArg(const BaseRef &arg, const AnfNodePtr &node,
                                                 std::vector<tensor::TensorPtr> *flatten_tensors) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(flatten_tensors);
  MS_LOG(DEBUG) << "Dynamic sequence node:" << node->fullname_with_scope() << " abs:" << node->abstract()->ToString();
  if (!utils::isa<ValuePtr>(arg)) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid input for dynamic sequence node:"
                               << node->DebugString();
  }
  auto value = utils::cast<ValuePtr>(arg);
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueSequence>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid value:" << value->ToString()
                               << " for dynamic sequence node:" << node->DebugString();
  }
  const auto &tensor = AnfAlgo::SequenceToTensor(value);
  flatten_tensors->emplace_back(tensor);
}

void AnfRuntimeAlgorithm::FlattenInputArg(const BaseRef &arg, const AnfNodePtr &node,
                                          std::vector<tensor::TensorPtr> *flatten_tensors) {
  MS_EXCEPTION_IF_NULL(flatten_tensors);
  if (node != nullptr && node->abstract() != nullptr && common::AnfAlgo::IsDynamicSequence(node)) {
    FlattenDynamicInputArg(arg, node, flatten_tensors);
    return;
  }

  if (utils::isa<tensor::TensorPyWrapperBase>(arg)) {
    auto base_tensor = utils::cast<tensor::TensorPyWrapperBasePtr>(arg)->GetTensorWrapper();
    tensor::TensorPtr tensor = utils::cast<tensor::TensorPtr>(base_tensor);
    if (tensor == nullptr) {
      tensor = std::make_shared<tensor::Tensor>(*base_tensor);
    }
    (void)flatten_tensors->emplace_back(tensor);
  } else if (utils::isa<tensor::Tensor>(arg)) {
    (void)flatten_tensors->emplace_back(utils::cast<tensor::TensorPtr>(arg));
  } else if (utils::isa<tensor::Tensor>(arg)) {
    (void)flatten_tensors->emplace_back(std::make_shared<tensor::Tensor>(*utils::cast<tensor::TensorPtr>(arg)));
  } else if (utils::isa<Scalar>(arg)) {
    (void)flatten_tensors->emplace_back(ScalarToTensor(utils::cast<ScalarPtr>(arg)));
  } else if (utils::isa<Monad>(arg)) {
    // If value is a monad, replace it with an unused tensor.
    flatten_tensors->push_back(std::make_shared<tensor::Tensor>(int64_t(0), kBool));
  } else if (utils::isa<ValueSequencePtr>(arg)) {
    auto value_sequence = utils::cast<ValueSequencePtr>(arg);
    MS_EXCEPTION_IF_NULL(value_sequence);
    auto sequence_value = value_sequence->value();
    for (auto &value : sequence_value) {
      FlattenInputArg(value, node, flatten_tensors);
    }
  } else if (utils::isa<ValueDictionaryPtr>(arg)) {
    auto value_dict = utils::cast<ValueDictionaryPtr>(arg);
    MS_EXCEPTION_IF_NULL(value_dict);
    auto dict_value = value_dict->value();
    for (auto &iter : dict_value) {
      FlattenInputArg(iter.second, node, flatten_tensors);
    }
  } else if (utils::isa<tensor::COOTensorPtr>(arg)) {
    auto coo_tensor = utils::cast<tensor::COOTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(coo_tensor);
    for (size_t i = 0; i < coo_tensor->GetTensorLength(); ++i) {
      (void)flatten_tensors->emplace_back(coo_tensor->GetTensorAt(i));
    }
  } else if (utils::isa<tensor::CSRTensorPtr>(arg)) {
    auto csr_tensor = utils::cast<tensor::CSRTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(csr_tensor);
    for (size_t i = 0; i < csr_tensor->GetTensorLength(); ++i) {
      (void)flatten_tensors->emplace_back(csr_tensor->GetTensorAt(i));
    }
  } else if (utils::isa<VectorRefPtr>(arg)) {
    const auto &args_new = utils::cast<VectorRef>(arg);
    for (const auto &arg_new : args_new) {
      FlattenInputArg(arg_new, node, flatten_tensors);
    }
  } else {
    MS_LOG(INTERNAL_EXCEPTION)
      << "#dmsg#Runtime error info:#dmsg#The value input to flatten tensor not supported for type " << arg.ToString();
  }
}

void AnfRuntimeAlgorithm::UpdateValueNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return;
  }
  const auto &value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueSequence>()) {
    return;
  }
  const auto &value_sequence = value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_sequence);
  std::vector<abstract::AbstractBasePtr> abstract_list;
  for (const auto &sub_value : value_sequence->value()) {
    MS_EXCEPTION_IF_NULL(sub_value);
    if (sub_value->isa<Scalar>()) {
      auto abstract = std::make_shared<abstract::AbstractScalar>(sub_value->type());
      (void)abstract_list.emplace_back(abstract);
    } else if (sub_value->isa<tensor::Tensor>()) {
      const auto &tensor = sub_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto abstract = std::make_shared<abstract::AbstractTensor>(tensor->Dtype(), tensor->shape());
      (void)abstract_list.emplace_back(abstract);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid value:" << sub_value->ToString()
                                        << " in dynamic sequence value node:" << node->DebugString();
    }
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_LOG(INFO) << "Set abstract for node:" << node->DebugString() << "from:" << node->abstract()->ToString()
               << " to:" << abstract_tuple->ToString();
  node->set_abstract(abstract_tuple);
}

bool AnfRuntimeAlgorithm::HasSelectKernelBuildInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return false;
  }
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return false;
  }
  return true;
}

bool AnfRuntimeAlgorithm::NeedEraseCache(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(kRandomCache)) {
    return false;
  }
  auto random_cache_value = prim->GetAttr(kRandomCache);
  MS_EXCEPTION_IF_NULL(random_cache_value);
  return !GetValue<bool>(random_cache_value);
}

abstract::AbstractBasePtr AnfRuntimeAlgorithm::GetNodeAbstractByIndex(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract = node->abstract();
  if (abstract == nullptr) {
    return abstract;
  }

  // Return output abstract directly for : 1.not sequence type, 2.dynamic sequence type, 3.real tuple/list type.
  if (!abstract->isa<abstract::AbstractSequence>() || common::AnfAlgo::IsDynamicSequence(node) ||
      (node->isa<CNode>() && !mindspore::AnfAlgo::GetOutputKernelObjectTypes(node).empty() &&
       (mindspore::session::AnfRuntimeAlgorithm::GetOutputKernelObjectType(node, 0) ==
        kernel::KernelObjectType::TUPLE))) {
    MS_EXCEPTION_IF_CHECK_FAIL((index == 0), "Cannot get " + std::to_string(index) + " child abstract from " +
                                               abstract->ToString() + " in node:" + node->fullname_with_scope());
    return abstract;
  }

  // Return element abstract by index for tuple type.
  const auto &abstract_tuple = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  const auto &elements = abstract_tuple->elements();
  if (elements.size() <= index) {
    const auto sub_abstract = common::AnfAlgo::FetchAbstractByIndex(node->abstract(), index);
    return sub_abstract;
  }
  return elements[index];
}

ValueNodePtr AnfRuntimeAlgorithm::CreateTypeIdValueNodeToKernelGraph(const FuncGraphPtr &func_graph, TypeId data_type) {
  auto type_id_value_node = NewValueNode(static_cast<int64_t>(data_type));
  auto type_id_value = std::make_shared<Int64Imm>(static_cast<int64_t>(data_type));
  type_id_value_node->set_abstract(type_id_value->ToAbstract());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  type_id_value_node = kernel_graph->NewValueNode(type_id_value_node);
  kernel_graph->AddValueNodeToGraph(type_id_value_node);
  return type_id_value_node;
}

ValueNodePtr AnfRuntimeAlgorithm::CreateTypeIdValueNodeToFuncGraph(const FuncGraphPtr &func_graph, TypeId data_type) {
  auto type_id_value_node = NewValueNode(static_cast<int64_t>(data_type));
  auto type_id_value = std::make_shared<Int64Imm>(static_cast<int64_t>(data_type));
  type_id_value_node->set_abstract(type_id_value->ToAbstract());
  return type_id_value_node;
}

bool AnfRuntimeAlgorithm::IsNoRealKernelGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    if (AnfUtils::IsRealKernel(node)) {
      return false;
    }
  }
  return true;
}

bool AnfRuntimeAlgorithm::IsGraphOutputValueNodeOrParameterForCompile(const AnfNodePtr &graph_output) {
  MS_EXCEPTION_IF_NULL(graph_output);
  if (graph_output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to compile.";
    return true;
  }

  if (graph_output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to compile.";
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsGraphOutputValueNodeOrParameter(const AnfNodePtr &graph_output, const VectorRef &args,
                                                            VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(graph_output);
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    VectorRef output_tmp;
    ValuePtr value = GetValueNode(graph_output);
    TensorValueToVector(value, &output_tmp);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      outputs->emplace_back(output_tmp);
    } else if (value->isa<tensor::Tensor>() || value->isa<Scalar>()) {
      *outputs = output_tmp;
    } else {
      MS_LOG(INFO) << "Graph output is empty!";
    }
    return true;
  }

  if (graph_output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    // Find the right parameter as ret_val.
    auto func_graph = graph_output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if (args.size() != params.size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Input size " << args.size()
                                 << " is not equal to graph input size " << params.size();
    }

    auto it = std::find(params.begin(), params.end(), graph_output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter, it should be found in graph parameters";
    }
    size_t index = static_cast<size_t>(it - params.cbegin());
    if (index >= args.size()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size();
    }

    outputs->emplace_back(args[index]);
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_mod = GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // Search for the ignore list if kernelmod has ignore list.
  std::vector<size_t> launch_ignored_input_idx = kernel_mod->GetLaunchIgnoredInputAddressIdx();
  if (!launch_ignored_input_idx.empty()) {
    if (std::find(launch_ignored_input_idx.begin(), launch_ignored_input_idx.end(), input_idx) !=
        launch_ignored_input_idx.end()) {
      return true;
    }
    return false;
  }

  // The new ignore input cannot be dumped, so it is not ignored when dump.
  static bool is_enable_dump = !common::GetEnv(kMindsporeDumpConfig).empty();
  if (is_enable_dump) {
    return false;
  }

  auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  auto cnode = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->abstract());
  const auto &input_type = cnode->abstract()->BuildType();
  // Tensor or tuple of tensor should not be ignored.
  if (input_type->type_id() == kObjectTypeTensorType) {
    return false;
  }

  if (input_type->type_id() == kObjectTypeTuple) {
    auto type = input_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(type);
    if (type->dynamic_len()) {
      return false;
    }
    const auto &elements = type->elements();
    if (!elements.empty() && elements[0]->type_id() == kObjectTypeTensorType) {
      return false;
    }
  }

  if (input_type->type_id() == kObjectTypeList) {
    auto type = input_type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(type);
    if (type->dynamic_len()) {
      return false;
    }
    const auto &elements = type->elements();
    if (!elements.empty() && elements[0]->type_id() == kObjectTypeTensorType) {
      return false;
    }
  }

  return true;
}

std::string AnfRuntimeAlgorithm::GetValueByDeviceAddress(DeviceAddress *const device_address, size_t element_num) {
  if (device_address == nullptr || device_address->GetPtr() == nullptr) {
    return "";
  }
  size_t size = device_address->GetSize();
  std::string value;
  char *buf = new char[size];
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    delete[] buf;
    buf = reinterpret_cast<char *>(device_address->GetMutablePtr());
  }
  device_address->SyncDeviceToHost(size, buf);
  auto is_vaild_index = [element_num](size_t index, size_t total) { return index < total && index < element_num; };
  std::string result;
  if (device_address->type_id() == TypeId::kNumberTypeInt32) {
    for (size_t i = 0; is_vaild_index(i, size / sizeof(int)); ++i) {
      value += std::to_string((reinterpret_cast<int *>(buf))[i]);
      value += ", ";
    }
    result = " type int, value:" + value;
  } else if (device_address->type_id() == TypeId::kNumberTypeInt64) {
    for (size_t i = 0; is_vaild_index(i, size / sizeof(int64_t)); ++i) {
      value += std::to_string((reinterpret_cast<int64_t *>(buf))[i]);
      value += ", ";
    }
    result = " type int64, value:" + value;
  } else if (device_address->type_id() == TypeId::kNumberTypeFloat32) {
    for (size_t i = 0; is_vaild_index(i, size / sizeof(float)); ++i) {
      value += std::to_string(reinterpret_cast<float *>((reinterpret_cast<void *>(buf)))[i]);
      value += ", ";
    }
    result = " type float32, value:" + value;
  } else if (device_address->type_id() == TypeId::kNumberTypeFloat64) {
    for (size_t i = 0; is_vaild_index(i, size / sizeof(double)); ++i) {
      value += std::to_string((reinterpret_cast<double *>(reinterpret_cast<void *>(buf)))[i]);
      value += ", ";
    }
    result = " type floa64, value:" + value;
  } else if (device_address->type_id() == TypeId::kNumberTypeBool) {
    for (size_t i = 0; is_vaild_index(i, size); ++i) {
      value += std::to_string((reinterpret_cast<bool *>(buf))[i]);
      value += ", ";
    }
    result = " type bool, value:" + value;
  } else if (device_address->type_id() == TypeId::kNumberTypeFloat16) {
    constexpr size_t kFloat16TypeSize = 2;
    for (size_t i = 0; is_vaild_index(i, size / kFloat16TypeSize); ++i) {
      float fp32 = 0;
      device::HalfToFloat(&fp32, buf + i * kFloat16TypeSize, 1);
      value += std::to_string(fp32);
      value += ", ";
    }
    result = " type float16, value:" + value;
  }
  if (device_address->GetDeviceType() != device::DeviceType::kCPU) {
    delete[] buf;
  }
  return result;
}

void AnfRuntimeAlgorithm::SetKernelObjectTypeBuildInfo(
  const AnfNodePtr &kernel_node, const std::vector<kernel::KernelObjectType> &input_kernel_object_types,
  const std::vector<kernel::KernelObjectType> &output_kernel_object_types) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (kernel_node->kernel_info() == nullptr) {
    kernel_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  }
  if (!kernel_node->kernel_info()->has_build_info()) {
    AnfAlgo::SetSelectKernelBuildInfo(std::make_shared<kernel::KernelBuildInfo>(), kernel_node.get());
  }

  MS_LOG(DEBUG) << kernel_node->fullname_with_scope() << " input kernel object type is: " << input_kernel_object_types
                << ", output kernel object type is: " << output_kernel_object_types;
  auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  kernel_build_info->SetOutputsKernelObjectType(output_kernel_object_types);
  kernel_build_info->SetInputsKernelObjectType(input_kernel_object_types);
}

void AnfRuntimeAlgorithm::SetKernelObjectTypeBuildInfo(
  const AnfNodePtr &kernel_node, const std::vector<kernel::KernelObjectType> &input_kernel_object_types,
  const std::vector<kernel::KernelObjectType> &output_kernel_object_types,
  const std::vector<kernel::KernelObjectType> &output_elements_kernel_object_types) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (kernel_node->kernel_info() == nullptr) {
    kernel_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  }
  if (!kernel_node->kernel_info()->has_build_info()) {
    AnfAlgo::SetSelectKernelBuildInfo(std::make_shared<kernel::KernelBuildInfo>(), kernel_node.get());
  }

  MS_LOG(DEBUG) << kernel_node->fullname_with_scope() << " input kernel object type is: " << input_kernel_object_types
                << ", output kernel object type is: " << output_kernel_object_types
                << ", output elements kernel object type is: " << output_elements_kernel_object_types;
  auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  kernel_build_info->SetOutputsKernelObjectType(output_kernel_object_types);
  kernel_build_info->SetInputsKernelObjectType(input_kernel_object_types);
  kernel_build_info->SetOutputElementsKernelObjectType(output_elements_kernel_object_types);
}

static std::vector<KernelObjectType> CalInputKernelObjectTypes(const AnfNodePtr &kernel_node,
                                                               const kernel::KernelAttr &selected_kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_input_object_types = kernel::GetInputObjectTypeListFromKernelAttr(selected_kernel_attr);
  auto input_object_types = AnfAlgo::GetAllInputObjectType(kernel_node);
  return kernel::CalKernelObjectTypes(input_object_types, selected_input_object_types,
                                      selected_kernel_attr.GetAllSame(), selected_kernel_attr.GetSkipCheck());
}

static bool HasOutputElementsKernelObjectType(const std::vector<kernel::KernelObjectType> &output_kernel_object_types) {
  return output_kernel_object_types.size() == 1 &&
         output_kernel_object_types[0] == kernel::KernelObjectType::TUPLE_UNFOLD;
}

static std::vector<KernelObjectType> CalOutputKernelObjectTypes(const AnfNodePtr &kernel_node,
                                                                const kernel::KernelAttr &selected_kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_output_object_types = kernel::GetOutputObjectTypeListFromKernelAttr(selected_kernel_attr);
  auto output_object_types = AnfAlgo::GetAllOutputObjectType(kernel_node);
  return kernel::CalKernelObjectTypes(output_object_types, selected_output_object_types,
                                      selected_kernel_attr.GetAllSame(), selected_kernel_attr.GetSkipCheck());
}

void AnfRuntimeAlgorithm::SetKernelObjectTypeWithSelectedAttr(const CNodePtr &kernel_node,
                                                              const kernel::KernelAttr &selected_kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<kernel::KernelObjectType> input_kernel_object_types;
  if (common::AnfAlgo::HasNodeAttr(kInputRealTuple, kernel_node)) {
    input_kernel_object_types = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(kernel_node));
  } else {
    input_kernel_object_types = CalInputKernelObjectTypes(kernel_node, selected_kernel_attr);
  }

  std::vector<KernelObjectType> output_kernel_object_types;
  if (common::AnfAlgo::HasNodeAttr(kOutputRealTuple, kernel_node)) {
    output_kernel_object_types = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllOutputObjectType(kernel_node));
  } else {
    output_kernel_object_types = CalOutputKernelObjectTypes(kernel_node, selected_kernel_attr);
  }

  std::vector<KernelObjectType> output_element_object_types;
  if (HasOutputElementsKernelObjectType(output_kernel_object_types)) {
    output_element_object_types = CalOutputElementObjectTypes(kernel_node, selected_kernel_attr);
  }
  MS_LOG(DEBUG) << "Set kernel object type:" << output_kernel_object_types
                << " for node:" << kernel_node->fullname_with_scope();
  SetKernelObjectTypeBuildInfo(kernel_node, input_kernel_object_types, output_kernel_object_types,
                               output_element_object_types);
}

static bool IsObjectTypeStrictlyMatched(const std::vector<TypeId> &object_types,
                                        const std::vector<kernel::DataType> &kernel_data_types) {
  if (object_types.size() != kernel_data_types.size()) {
    return false;
  }

  for (size_t i = 0; i < object_types.size(); i++) {
    // For optional input, the real input object type can be a None.
    if ((object_types[i] != kernel_data_types[i].object_type) &&
        !(object_types[i] == kMetaTypeNone && kernel_data_types[i].is_optional)) {
      return false;
    }
  }

  return true;
}

static bool IsObjectTypeWeaklyMatched(const std::vector<TypeId> &object_types,
                                      const std::vector<kernel::DataType> &kernel_data_types, bool all_same,
                                      size_t element_num) {
  // 1. The size equal can trigger the kernel object backoff(For example Reshape op).
  if (object_types.size() == kernel_data_types.size()) {
    return true;
  }

  // 2. AllSame is the tupleUnfold type(For example Split/Addn op).
  if (all_same) {
    return true;
  }

  // 3. Multiple outputs are expanded in the kernel attr(For example BatchNorm op).
  if (kernel_data_types.size() == element_num) {
    return true;
  }

  return false;
}

bool AnfRuntimeAlgorithm::SelectKernelByObjectType(const CNodePtr &kernel_node,
                                                   const std::vector<kernel::KernelAttr> &registered_kernel_attrs,
                                                   std::vector<kernel::KernelAttr> *selected_kernel_attrs) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_attrs);
  const auto &inputs_object_types = GetAllInputObjectType(kernel_node);
  const auto &output_object_types = GetAllOutputObjectType(kernel_node);

  // 1. Try match all object type firstly.
  for (auto &cur_kernel_attr : registered_kernel_attrs) {
    const auto &[input_data_types, output_data_types] = GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    if (IsObjectTypeStrictlyMatched(inputs_object_types, input_data_types) &&
        IsObjectTypeStrictlyMatched(output_object_types, output_data_types)) {
      (void)selected_kernel_attrs->emplace_back(cur_kernel_attr);
    }
  }
  if (!selected_kernel_attrs->empty()) {
    return true;
  }

  // 2. Precise matching failed, try fuzzy one again.
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  auto output_num = GetOutputElementNum(kernel_node);
  for (auto &cur_kernel_attr : registered_kernel_attrs) {
    const auto &[input_data_types, output_data_types] = GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    auto all_same = cur_kernel_attr.GetAllSame();
    if (IsObjectTypeWeaklyMatched(inputs_object_types, input_data_types, all_same, input_num) &&
        IsObjectTypeWeaklyMatched(output_object_types, output_data_types, all_same, output_num)) {
      (void)selected_kernel_attrs->emplace_back(cur_kernel_attr);
    }
  }

  return (!selected_kernel_attrs->empty());
}

kernel::KernelAttr AnfRuntimeAlgorithm::GetKernelAttrFromNode(const AnfNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto build_info = GetSelectKernelBuildInfo(kernel_node);
  return GetKernelAttrFromBuildInfo(build_info);
}
}  // namespace mindspore::session
