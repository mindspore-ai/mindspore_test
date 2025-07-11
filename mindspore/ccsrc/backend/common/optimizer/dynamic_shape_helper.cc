/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "backend/common/optimizer/dynamic_shape_helper.h"

#include <memory>
#include <algorithm>
#include <stack>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/ms_device_shape_transfer.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/framework_utils.h"
#include "ops/op_def.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "debug/profiler/profiler.h"
#include "ir/anf.h"
#include "ir/functor.h"
#include "backend/operator/ops_backend_infer_function.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace opt::dynamic_shape {
namespace {
constexpr int64_t kInvalidShape = -2;

void InferShapeForNopNode(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!common::AnfAlgo::IsNopNode(input_node)) {
    MS_LOG(INFO) << "Input node is not a nop node, no need infer.";
    return;
  }
  if (!common::AnfAlgo::IsNeedSkipNopOpExecution(input_node)) {
    MS_LOG(INFO) << "The Nop node need execution, no need the InferShapeForNopNode.";
    return;
  }
  MS_LOG(INFO) << "Infer shape for nop node.";
  std::stack<AnfNodePtr> nop_road;
  nop_road.push(input_node);

  auto in_node = input_node;
  while (true) {
    auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(in_node, 0);
    in_node = input_node_with_idx.first;
    MS_EXCEPTION_IF_NULL(in_node);
    if (common::AnfAlgo::IsNopNode(in_node)) {
      nop_road.push(in_node);
    } else {
      break;
    }
  }

  while (!nop_road.empty()) {
    auto nop_node = nop_road.top();
    MS_EXCEPTION_IF_NULL(nop_node);
    AnfAlgo::InferShape(nop_node->cast<CNodePtr>());
    nop_road.pop();
  }
}

tensor::TensorPtr GetDependValueTensor(const AnfNodePtr &node, size_t i,
                                       const std::pair<AnfNodePtr, size_t> &input_node_with_index, bool skip_nop_node,
                                       void *args) {
  MS_LOG(EXCEPTION) << "Not supported.";
}

template <typename T>
abstract::AbstractBasePtrList MakeElemsByTensorValue(void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  T *tensor_data = static_cast<T *>(data);
  AbstractBasePtrList elems;
  for (size_t i = 0; i < size; i++) {
    auto scalar = std::make_shared<abstract::AbstractScalar>(tensor_data[i]);
    (void)elems.emplace_back(scalar);
  }
  return elems;
}

abstract::AbstractBasePtr MakeNewAbstract(const AnfNodePtr &input, const tensor::TensorPtr &depended_value,
                                          const size_t &input_index) {
  MS_LOG(EXCEPTION) << "Not supported.";
}

void InferShapeForPrimitive(const CNodePtr &cnode, const PrimitivePtr &primitive,
                            const AbstractBasePtrList &args_spec_list, bool has_py_execute_data) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!has_py_execute_data && !IsPrimitiveCNode(cnode, prim::kPrimPyExecute)) {
    // Pynative mode is rely on the origin abstract of cnode, so cannot modify the abstract inplace, clone from old
    // abstract instead.
    opt::CppInferShape(primitive, args_spec_list, cnode);
  }
}

void InferShape(const CNodePtr &cnode, std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(DEBUG) << "InferShape start, node:" << cnode->fullname_with_scope();
  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(cnode);

  depend_tensor_map->clear();
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  bool skip_nop_node = !context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  bool has_py_execute_data = false;
  kernel::PyExecuteOutputUserDataPtr list_user_data = nullptr;
  std::vector<size_t> list_start_index;
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i, false);
    auto real_input = input_node_with_index.first;
    auto real_input_index = input_node_with_index.second;

    MS_EXCEPTION_IF_NULL(real_input);
    if (skip_nop_node) {
      InferShapeForNopNode(real_input);
    }

    if (depend_list.find(i) != depend_list.end()) {
      auto depended_value = GetDependValueTensor(cnode, i, input_node_with_index, skip_nop_node, args);
      auto ret2 = depend_tensor_map->try_emplace(i, depended_value);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }

      auto updated_abs = MakeNewAbstract(real_input, depended_value, real_input_index);
      MS_EXCEPTION_IF_NULL(updated_abs);
      MS_EXCEPTION_IF_NULL(real_input);
      MS_EXCEPTION_IF_NULL(real_input->abstract());
      if (updated_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
        has_py_execute_data = true;
        if (IsPrimitiveCNode(real_input, prim::kPrimPyExecute) &&
            real_input->abstract()->isa<abstract::AbstractSequence>()) {
          auto updated_abs_user_data = updated_abs->user_data<kernel::PyExecuteOutputUserData>();
          if (list_user_data == nullptr || list_user_data != updated_abs_user_data) {
            list_start_index.push_back(i);
            list_user_data = updated_abs_user_data;
          }
        }
      }
      (void)args_spec_list.emplace_back(updated_abs);
    } else {
      auto abs = real_input->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      MS_LOG(DEBUG) << "Real input node:" << real_input->DebugString() << " abs:" << abs->ToString()
                    << " index:" << real_input_index;
      if (abs->isa<abstract::AbstractSequence>() && !AnfAlgo::IsRealSquenceOutput(real_input)) {
        auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
        MS_EXCEPTION_IF_NULL(abs_seq);
        MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abs_seq->elements().size()), "Index is out of range.");
        auto abs_index = abs_seq->elements()[real_input_index];
        (void)args_spec_list.emplace_back(abs_index);
      } else {
        (void)args_spec_list.emplace_back(abs);
      }
    }
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  if (auto primitive = GetValueNode<PrimitivePtr>(inputs[0])) {
    MS_EXCEPTION_IF_NULL(primitive);
    (void)primitive->AddAttr(kAttrListStartIndex, MakeValue(list_start_index));
    InferShapeForPrimitive(cnode, primitive, args_spec_list, has_py_execute_data);
  } else {
    MS_LOG_WITH_NODE(EXCEPTION, inputs[0])
      << "The first input of the cnode should be either a primitive or a function graph, but get: "
      << inputs[0]->fullname_with_scope();
  }
  MS_LOG(DEBUG) << "InferShape end, node:" << cnode->fullname_with_scope();
}

inline bool IsCpuKernelMod(kernel::KernelModType kernel_mod_type) {
  return kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod;
}
}  // namespace

BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferInner,
                                     op_name, true);
  if (primitive->HasAttr(kAttrInferShapeFunctor)) {
    auto functor = primitive->GetAttr(kAttrInferShapeFunctor)->cast<InferShapeFunctorPtr>();
    MS_EXCEPTION_IF_NULL(functor);
    return functor->InferShape(input_args);
  }
  auto shape_optional = abstract::InferShapeByFuncImpl(primitive, input_args, false);
  if (shape_optional.has_value()) {
    return shape_optional.value();
  }

  // The old register map for InferShape will be deleted in the future.
  auto infer_impl = abstract::GetBackendPrimitiveInferImpl(primitive);
  if (infer_impl.has_value()) {
    auto infer = infer_impl.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShape(primitive, input_args);
    }
  }
  MS_LOG(EXCEPTION) << "The InferShape function of [" << op_name << "] is not defined.";
}

void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                             const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(base_shape);
  size_t output_num = output_kernel_tensors.size();
  if (output_num > 1) {
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    const auto &shapes = sequence_shape->shape();
    if (shapes.size() != output_num) {
      MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << output_num
                        << ", but got: " << shapes.size();
    }
    for (size_t i = 0; i < output_num; i++) {
      const auto &kernel_tensor = output_kernel_tensors[i];
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->SetShape(shapes[i]);
    }
  } else if (output_num == 1) {
    const auto &kernel_tensor = output_kernel_tensors[0];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    if ((kernel_tensor->type_id() != kObjectTypeTuple && kernel_tensor->type_id() != kObjectTypeList) &&
        sequence_shape != nullptr) {
      // For the operator prototype whose output is of type Tuple, the back-end operator is expanded as Tensors, and for
      // single-output scenarios, the InferShape result is TupleShape, and the back-end needs to expand it to
      // TensorShape. For example, the output of the split operator is only a Tensor scene.
      const auto &shapes = sequence_shape->shape();
      if (shapes.size() != 1) {
        MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << 1 << ", but got: " << shapes.size();
      }

      kernel_tensor->SetShape(shapes[0]);
    } else {
      kernel_tensor->SetShape(base_shape);
    }
  }
}

abstract::AbstractBasePtr InferShapeAndType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  auto infer_impl = abstract::GetBackendPrimitiveInferImpl(primitive);
  if (infer_impl.has_value()) {
    auto infer = infer_impl.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShapeAndType(nullptr, primitive, input_args);
    }
  }
  MS_LOG(EXCEPTION) << "The InferShape function of [" << op_name << "] is not defined.";
}

void UpdateKernelTensorType(const TypePtr &type, const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(type);
  if (output_kernel_tensors.size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid output size:" << output_kernel_tensors.size();
  }

  const auto &kernel_tensor = output_kernel_tensors[0];
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->SetType(type);
}

void InferOp(const CNodePtr &cnode, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  kernel::KernelArgs kernel_args;
  MS_LOG(DEBUG) << "infer shape for node:" << cnode->fullname_with_scope();
  InferShape(cnode, &kernel_args.depend_tensor_map, args);
  auto kernel_mod_type = kernel_mod->GetKernelModType();
  auto update = kernel::AbstractArgsFromCNode(cnode);
  update.depend_tensor_map = std::move(kernel_args.depend_tensor_map);
  kernel::SetInputsByDependMap(update.depend_tensor_map, &update.inputs, IsCpuKernelMod(kernel_mod_type));
  kernel::SetArgsToCNode(cnode, update);
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
