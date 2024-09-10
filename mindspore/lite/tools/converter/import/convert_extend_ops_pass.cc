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

#define USE_DEPRECATED_API
#include "tools/converter/import/convert_extend_ops_pass.h"
#include <unordered_map>
#include <memory>
#include <vector>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameSumExtPatternName = "SumExtPatternName";

AnfNodePtr CreateCastNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                          const TypeId &dst_type_id) {
  auto cast_prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_CHECK_TRUE_MSG(cast_prim != nullptr, nullptr, "create Primitive cast failed.");
  cast_prim->AddAttr("input_names", MakeValue(std::vector<std::string>{"x", "dst_type"}));
  cast_prim->AddAttr("output_names", MakeValue(std::vector<std::string>{"output"}));
  cast_prim->AddAttr("primitive_function", MakeValue<bool>(true));

  auto dst_type = std::make_shared<Int64Imm>(static_cast<int64_t>(dst_type_id));
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(cast_prim), input, NewValueNode(dst_type)};
  auto cast_shape = input->Shape();
  auto cast_input = func_graph->NewCNode(cast_inputs);
  MS_CHECK_TRUE_MSG(cast_input != nullptr, nullptr, "create CNode cast failed.");
  auto cast_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(dst_type_id), cast_shape);
  cast_input->set_abstract(cast_abs);
  cast_input->set_scope(input->scope());

  return cast_input;
}

TypeId GetSingleNodeOutputTypeId(const mindspore::AnfNodePtr &node) {
  TypePtr type = node->Type();
  if (node->isa<CNode>()) {
    auto input0 = node->cast<CNodePtr>()->input(0);
    auto input0_value_node = dyn_cast_ptr<ValueNode>(input0);
    if (input0_value_node != nullptr && input0_value_node->value()->isa<Primitive>()) {
      const auto &abs = node->abstract();
      MS_CHECK_TRUE_MSG(abs != nullptr, kTypeUnknown, "get abstract from CNode failed.");
      type = abs->BuildType();
    }
  }

  MS_CHECK_TRUE_MSG(type != nullptr, kTypeUnknown, "type is nullptr.");
  if (type->isa<TensorType>()) {
    const auto &tensor_type = type->cast<TensorTypePtr>();
    MS_CHECK_TRUE_MSG(tensor_type != nullptr, kTypeUnknown, "cast TensorType failed.");
    const auto &element = tensor_type->element();
    return element->type_id();
  } else {
    return type->type_id();
  }
}

AnfNodePtr ReduceExtendGetCastInputByDtype(const FuncGraphPtr &func_graph, const mindspore::CNodePtr &cnode) {
  auto input = cnode->input(kInputIndexOne);
  auto dtype = cnode->input(kInputIndexFour);

  auto src_type_id = GetSingleNodeOutputTypeId(input);
  MS_CHECK_TRUE_MSG(src_type_id != kTypeUnknown, nullptr, "get src_type_id failed.");
  auto dtype_value_node_ptr = dtype->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(dtype_value_node_ptr != nullptr, nullptr, "dtype cannot be converted to a ValueNode.");
  auto dtype_value_ptr = utils::cast<ValuePtr>(dtype_value_node_ptr->value());
  if (dtype_value_ptr->isa<None>()) {
    const std::set<TypeId> kIntergralSet = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                            kNumberTypeInt32};
    if (kIntergralSet.find(src_type_id) != kIntergralSet.end()) {
      MS_LOG(INFO) << "For SumExt, when dtype of 'input' is [bool, uint8, int8, int16, int32, int64] and 'dtype' is "
                   << "None, then dtype of output will be int64.";
      return CreateCastNode(func_graph, input, kNumberTypeInt64);
    }
    return input;
  }

  auto dst_type_id = static_cast<TypeId>(GetValue<int64_t>(dtype_value_ptr));
  return dst_type_id != src_type_id ? CreateCastNode(func_graph, input, dst_type_id) : input;
}

AnfNodePtr ConvertSumExtPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto sum_ext_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sum_ext_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(sum_ext_cnode->size() == kInputSizeFive, nullptr);
  if (IsMarkedTrainOp(sum_ext_cnode)) {
    return nullptr;
  }
  if (!CheckPrimitiveType(sum_ext_cnode, prim::kPrimSumExt)) {
    return nullptr;
  }

  auto input = ReduceExtendGetCastInputByDtype(func_graph, sum_ext_cnode);
  MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "input is invalid.");
  auto axis = sum_ext_cnode->input(kInputIndexTwo);
  auto axis_value_node_ptr = axis->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(axis_value_node_ptr != nullptr, nullptr, "axis cannot be converted to a ValueNode.");
  auto axis_value_ptr = utils::cast<ValuePtr>(axis_value_node_ptr->value());
  if (axis_value_ptr->isa<None>()) {
    auto new_axis_value = MakeValue<std::vector<int64_t>>({});
    auto new_axis = NewValueNode(new_axis_value);
    new_axis->set_scope(axis->scope());
    new_axis->set_abstract(new_axis_value->ToAbstract());
    axis = new_axis;
  }

  const auto &keep_dims = sum_ext_cnode->input(kInputIndexThree);
  auto keep_dims_value_node_ptr = keep_dims->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(keep_dims_value_node_ptr != nullptr, nullptr, "keep_dims cannot be converted to a ValueNode.");

  auto sum_ext_prim_value_node = sum_ext_cnode->input(kInputIndexZero)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(sum_ext_prim_value_node != nullptr, nullptr, "primitive cannot be converted to a ValueNode.");
  auto sum_ext_prim_value = sum_ext_prim_value_node->value();
  MS_CHECK_TRUE_MSG(sum_ext_prim_value != nullptr, nullptr, "primitive cannot get value.");
  auto sum_ext_prim = sum_ext_prim_value->cast<PrimitivePtr>();
  MS_CHECK_TRUE_MSG(sum_ext_prim != nullptr, nullptr, "primitive cannot cast to Primitive.");

  auto reduce_sum_prim = std::make_shared<Primitive>(prim::kPrimReduceSum->name());
  MS_CHECK_TRUE_MSG(reduce_sum_prim != nullptr, nullptr, "create Primitive ReduceSum failed.");
  (void)reduce_sum_prim->SetAttrs(sum_ext_prim->attrs());
  reduce_sum_prim->AddAttr("keep_dims", keep_dims_value_node_ptr->value());
  reduce_sum_prim->AddAttr("skip_mode", MakeValue<bool>(false));

  std::vector<AnfNodePtr> reduce_sum_inputs = {NewValueNode(reduce_sum_prim), input, axis};
  auto reduce_sum_node = func_graph->NewCNode(reduce_sum_inputs);
  MS_CHECK_TRUE_MSG(reduce_sum_node != nullptr, nullptr, "create CNode reduce_sum failed.");
  reduce_sum_node->set_abstract(sum_ext_cnode->abstract());

  return reduce_sum_node;
}
}  // namespace

VectorRef ConvertExtendOpsPass::DefineSumExtPattern() const {
  auto is_sum_ext = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSumExt>);
  MS_CHECK_TRUE_RET(is_sum_ext != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto axis = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(axis != nullptr, {});
  auto keep_dims = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(keep_dims != nullptr, {});
  auto dtype = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dtype != nullptr, {});
  VectorRef sum_ext_ref = VectorRef({is_sum_ext, input, axis, keep_dims, dtype});
  return sum_ext_ref;
}

std::unordered_map<std::string, VectorRef> ConvertExtendOpsPass::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameSumExtPatternName] = DefineSumExtPattern();
  return patterns;
}

using ConvertExtendOpsSubPass = AnfNodePtr (*)(const FuncGraphPtr &, const mindspore::AnfNodePtr &);

AnfNodePtr ConvertExtendOpsPass::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                         const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  static std::unordered_map<std::string, ConvertExtendOpsSubPass> sub_pass_map = {
    {kNameSumExtPatternName, ConvertSumExtPass},
  };

  if (sub_pass_map.find(pattern_name) != sub_pass_map.end()) {
    MS_LOG(INFO) << "The node " << node->fullname_with_scope() << " is matched pattern[" << pattern_name
                 << "] in ConvertExtendOpsPass.";
    return sub_pass_map.at(pattern_name)(func_graph, node);
  }
  return nullptr;
}
}  // namespace mindspore::opt
