/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/vmap.h"

#include <cstdint>
#include <memory>
#include <string>
#include "mindspore/ops/op_def/sequence_ops.h"
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/pipeline.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/pybind_api/api_register.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
void GenerateFuncGraphAllNone(const FuncGraphPtr &fg, const AnfNodePtr &prim, int64_t args_size,
                              int64_t tuple_elements_num, bool bind) {
  std::vector<AnfNodePtr> prim_output_cnode_inputs;
  (void)prim_output_cnode_inputs.emplace_back(prim);
  if (tuple_elements_num != 0) {
    auto val_in_param = fg->add_parameter();
    std::vector<AnfNodePtr> prim_inputs_cnode_inputs;
    (void)prim_inputs_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (int64_t i = 0; i < tuple_elements_num; ++i) {
      auto val_in_cnode = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), val_in_param, NewValueNode(i)});
      auto val_cnode =
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), val_in_cnode, NewValueNode(kValIndex)});
      (void)prim_inputs_cnode_inputs.emplace_back(val_cnode);
    }
    auto prim_inputs_cnode = fg->NewCNodeInOrder(prim_inputs_cnode_inputs);
    (void)prim_output_cnode_inputs.emplace_back(prim_inputs_cnode);
    args_size = args_size - tuple_elements_num;
  }

  for (int64_t i = 0; i < args_size; ++i) {
    auto val_in_param = fg->add_parameter();
    auto val_cnode =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), val_in_param, NewValueNode(kValIndex)});
    (void)prim_output_cnode_inputs.emplace_back(val_cnode);
  }

  auto prim_output_cnode = fg->NewCNodeInOrder(prim_output_cnode_inputs);
  const py::function bind_all_none_fn = python_adapter::GetPyFn(kVmapFunctionModelName, "vmap_bind_all_none");
  auto bind_all_none_fg = parse::ParsePythonCode(bind_all_none_fn);
  MS_EXCEPTION_IF_NULL(bind_all_none_fg);
  auto bind_all_none_cnode = fg->NewCNodeInOrder({NewValueNode(bind_all_none_fg), prim_output_cnode});
  if (bind) {
    auto output_cnode =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), NewValueNode(true), bind_all_none_cnode});
    fg->set_output(output_cnode);
    return;
  }
  fg->set_output(bind_all_none_cnode);
  return;
}

CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerBroadcastAxis(
  const AnfNodePtr &inputs, const AnfNodePtr &out_axis, const AnfNodePtr &axis_size,
  const AbstractBasePtr &inputs_abstract_elements_begin) const {
  std::vector<AnfNodePtr> value_cnode_inputs;
  (void)value_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)value_cnode_inputs.emplace_back(inputs);
  (void)value_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
  auto value_cnode = fg_->NewCNodeInOrder(value_cnode_inputs);
  std::vector<AnfNodePtr> dim_cnode_inputs;
  (void)dim_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)dim_cnode_inputs.emplace_back(inputs);
  (void)dim_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(1)));
  auto dim_cnode = fg_->NewCNodeInOrder(dim_cnode_inputs);

  std::vector<AnfNodePtr> sub_inputs_cnode_inputs;
  (void)sub_inputs_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto inputs_abstract_elements_begin_tuple = dyn_cast<abstract::AbstractTuple>(inputs_abstract_elements_begin);
  MS_EXCEPTION_IF_NULL(inputs_abstract_elements_begin_tuple);
  auto inputs_abstract_elements_begin_tuple_elements = inputs_abstract_elements_begin_tuple->elements();
  // inputs: ((x, y), None) -> ((x, None), (y, None)).
  int64_t begin_tuple_size = static_cast<int64_t>(inputs_abstract_elements_begin_tuple_elements.size());
  for (int64_t i = 0; i < begin_tuple_size; ++i) {
    std::vector<AnfNodePtr> cur_tuple_getitem_inputs;
    (void)cur_tuple_getitem_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)cur_tuple_getitem_inputs.emplace_back(value_cnode);
    (void)cur_tuple_getitem_inputs.emplace_back(NewValueNode(i));
    auto cur_value_cnode = fg_->NewCNodeInOrder(cur_tuple_getitem_inputs);
    std::vector<AnfNodePtr> cur_make_tuple_cnode_inputs;
    (void)cur_make_tuple_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    (void)cur_make_tuple_cnode_inputs.emplace_back(cur_value_cnode);
    (void)cur_make_tuple_cnode_inputs.emplace_back(dim_cnode);
    auto cur_make_tuple_cnode = fg_->NewCNodeInOrder(cur_make_tuple_cnode_inputs);
    (void)sub_inputs_cnode_inputs.emplace_back(cur_make_tuple_cnode);
  }
  auto sub_inputs_cnode = fg_->NewCNodeInOrder(sub_inputs_cnode_inputs);
  std::vector<AnfNodePtr> out_cnode_inputs;
  (void)out_cnode_inputs.emplace_back(NewValueNode(std::make_shared<VmapMatchOutAxis>("VmapMatchOutAxis")));
  (void)out_cnode_inputs.emplace_back(sub_inputs_cnode);
  (void)out_cnode_inputs.emplace_back(out_axis);
  (void)out_cnode_inputs.emplace_back(axis_size);
  return fg_->NewCNodeInOrder(out_cnode_inputs);
}

CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerSingleElement(
  const AnfNodePtr &inputs, const AnfNodePtr &out_axis, const AnfNodePtr &axis_size,
  const AbstractBasePtr &inputs_abstract_elements_end) const {
  std::vector<AnfNodePtr> value_cnode_inputs;
  (void)value_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)value_cnode_inputs.emplace_back(inputs);
  (void)value_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
  auto value_cnode = fg_->NewCNodeInOrder(value_cnode_inputs);
  std::vector<AnfNodePtr> out_cnode_inputs;
  if (inputs_abstract_elements_end->isa<abstract::AbstractNone>()) {
    const py::function broadcast_by_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "_broadcast_by_axis");
    auto broadcast_by_axis_fg = parse::ParsePythonCode(broadcast_by_axis);
    MS_EXCEPTION_IF_NULL(broadcast_by_axis_fg);
    (void)out_cnode_inputs.emplace_back(NewValueNode(broadcast_by_axis_fg));
    (void)out_cnode_inputs.emplace_back(value_cnode);
    (void)out_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
    (void)out_cnode_inputs.emplace_back(axis_size);
  } else {
    std::vector<AnfNodePtr> dim_cnode_inputs;
    (void)dim_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)dim_cnode_inputs.emplace_back(inputs);
    (void)dim_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(1)));
    auto dim_cnode = fg_->NewCNodeInOrder(dim_cnode_inputs);
    const py::function move_axis = python_adapter::GetPyFn(kNumpyModelName, "moveaxis");
    auto move_axis_fg = parse::ParsePythonCode(move_axis);
    MS_EXCEPTION_IF_NULL(move_axis_fg);
    (void)out_cnode_inputs.emplace_back(NewValueNode(move_axis_fg));
    (void)out_cnode_inputs.emplace_back(value_cnode);
    (void)out_cnode_inputs.emplace_back(dim_cnode);
    (void)out_cnode_inputs.emplace_back(out_axis);
  }
  return fg_->NewCNodeInOrder(out_cnode_inputs);
}

namespace {
AbstractBasePtrList GetOutAxesAbstractElements(const AbstractBasePtr &out_axes_abstract,
                                               size_t inputs_abstract_elements_size, bool is_out_axes_tuple) {
  AbstractBasePtrList out_axes_abstract_elements;
  if (!is_out_axes_tuple) {
    return out_axes_abstract_elements;
  }
  abstract::AbstractTuplePtr out_axes_abstract_tuple = dyn_cast<abstract::AbstractTuple>(out_axes_abstract);
  MS_EXCEPTION_IF_NULL(out_axes_abstract_tuple);
  out_axes_abstract_elements = out_axes_abstract_tuple->elements();
  if (out_axes_abstract_elements.size() != inputs_abstract_elements_size) {
    MS_LOG(EXCEPTION) << "The length of out_axes and inputs do not match. ";
  }
  return out_axes_abstract_elements;
}
}  // namespace

CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerAllTuple(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                                          const AnfNodePtr &axis_size,
                                                          const AbstractBasePtrList &inputs_abstract_elements,
                                                          const AbstractBasePtr &out_axes_abstract) const {
  bool is_out_axes_tuple = out_axes_abstract->isa<abstract::AbstractTuple>();
  auto inputs_abstract_elements_size = inputs_abstract_elements.size();
  AbstractBasePtrList out_axes_abstract_elements =
    GetOutAxesAbstractElements(out_axes_abstract, inputs_abstract_elements_size, is_out_axes_tuple);

  std::vector<AnfNodePtr> vals_out_tuple_cnode_inputs;
  (void)vals_out_tuple_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  constexpr size_t kEachInputsSize = 2;
  // inputs: (((x1, x1_axis), (x2, x2_axis)), ((y1, y2), y_axis), (z, z_axis))
  for (int64_t i = 0; i < static_cast<int64_t>(inputs_abstract_elements_size); ++i) {
    std::vector<AnfNodePtr> each_input_cnode_inputs;
    (void)each_input_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)each_input_cnode_inputs.emplace_back(inputs);
    (void)each_input_cnode_inputs.emplace_back(NewValueNode(i));
    auto each_input_cnode = fg_->NewCNodeInOrder(each_input_cnode_inputs);
    AnfNodePtr dst_cnode = nullptr;
    if (is_out_axes_tuple) {
      dst_cnode = fg_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), out_axis, NewValueNode(i)});
    } else {
      dst_cnode = out_axis;
    }
    auto each_input_abstract = inputs_abstract_elements[i];
    AbstractBasePtr dst_abstract = is_out_axes_tuple ? out_axes_abstract_elements[i] : out_axes_abstract;
    auto each_input_abstract_tuple = dyn_cast<abstract::AbstractTuple>(each_input_abstract);
    MS_EXCEPTION_IF_NULL(each_input_abstract_tuple);
    auto each_inputs_abstract_elements = each_input_abstract_tuple->elements();
    auto each_inputs_abstract_elements_size = each_inputs_abstract_elements.size();
    if (each_inputs_abstract_elements_size == 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "Each_inputs_abstract_elements_size is empty";
    }
    auto each_inputs_abstract_elements_begin = each_inputs_abstract_elements[0];
    if (each_inputs_abstract_elements_begin->isa<abstract::AbstractTuple>()) {
      auto each_inputs_abstract_elements_end = each_inputs_abstract_elements.back();
      if (each_inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
        // current each input: ((x1, x1_axis), (x2, x2_axis)).
        std::vector<AnfNodePtr> out_cnode_inputs;
        (void)out_cnode_inputs.emplace_back(NewValueNode(std::make_shared<VmapMatchOutAxis>("VmapMatchOutAxis")));
        (void)out_cnode_inputs.emplace_back(each_input_cnode);
        (void)out_cnode_inputs.emplace_back(dst_cnode);
        (void)out_cnode_inputs.emplace_back(axis_size);
        (void)vals_out_tuple_cnode_inputs.emplace_back(fg_->NewCNodeInOrder(out_cnode_inputs));
      } else {
        // current each input: ((y1, y2), y_axis).
        auto out_cnode = GenerateFuncGraphInnerBroadcastAxis(each_input_cnode, dst_cnode, axis_size,
                                                             each_inputs_abstract_elements_begin);
        (void)vals_out_tuple_cnode_inputs.emplace_back(out_cnode);
      }
    } else {
      // current each input: (z, z_axis).
      if (each_inputs_abstract_elements_size != kEachInputsSize) {
        MS_LOG(EXCEPTION) << "Each input with no tuple should have only two elements.";
      }
      auto val_cnode = fg_->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), each_input_cnode, NewValueNode(static_cast<int64_t>(0))});
      auto src_cnode = fg_->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), each_input_cnode, NewValueNode(static_cast<int64_t>(1))});
      auto src_abstract = each_inputs_abstract_elements[1];
      CNodePtr out_cnode = nullptr;
      if (src_abstract->isa<abstract::AbstractNone>() && !dst_abstract->isa<abstract::AbstractNone>()) {
        const py::function broadcast_by_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "_broadcast_by_axis");
        auto broadcast_by_axis_fg = parse::ParsePythonCode(broadcast_by_axis);
        MS_EXCEPTION_IF_NULL(broadcast_by_axis_fg);
        out_cnode = fg_->NewCNodeInOrder({NewValueNode(broadcast_by_axis_fg), val_cnode, dst_cnode, axis_size});
      } else if (!src_abstract->isa<abstract::AbstractNone>() && dst_abstract->isa<abstract::AbstractNone>()) {
        MS_LOG(EXCEPTION) << "It is invalid that source is not None and dst is None.";
      } else if (src_abstract->isa<abstract::AbstractNone>() && dst_abstract->isa<abstract::AbstractNone>()) {
        out_cnode = val_cnode;
      } else {
        const py::function move_axis = python_adapter::GetPyFn(kNumpyModelName, "moveaxis");
        auto move_axis_fg = parse::ParsePythonCode(move_axis);
        MS_EXCEPTION_IF_NULL(move_axis_fg);
        out_cnode = fg_->NewCNodeInOrder({NewValueNode(move_axis_fg), val_cnode, src_cnode, dst_cnode});
      }
      (void)vals_out_tuple_cnode_inputs.emplace_back(out_cnode);
    }
  }
  return fg_->NewCNodeInOrder(vals_out_tuple_cnode_inputs);
}

FuncGraphPtr VmapMatchOutAxis::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  auto args_abs_list_size = args_abs_list.size();
  constexpr size_t kMetaFGInputSize = 3;
  if (args_abs_list_size != kMetaFGInputSize) {
    MS_LOG(EXCEPTION) << "The number of inputs to VmapMatchOutAxis should be 3, but got " << args_abs_list_size << ".";
  }
  auto inputs_abstract = args_abs_list[kIndex0];
  auto out_axes_abstract = args_abs_list[kIndex1];
  auto axis_size_abstract = args_abs_list[kIndex2];
  MS_EXCEPTION_IF_NULL(inputs_abstract);
  MS_EXCEPTION_IF_NULL(out_axes_abstract);
  MS_EXCEPTION_IF_NULL(axis_size_abstract);

  if (!inputs_abstract->isa<abstract::AbstractTuple>() && !inputs_abstract->isa<abstract::AbstractList>()) {
    MS_LOG(EXCEPTION) << "The first input to VmapMatchOutAxis is vmap_inputs and should be a tuple or list, but got "
                      << inputs_abstract->ToString() << ".";
  }
  auto out_axes_abstract_value = out_axes_abstract->BuildValue();
  if (out_axes_abstract_value == nullptr || out_axes_abstract_value->ContainsValueAny()) {
    MS_LOG(EXCEPTION) << "The second input to VmapMatchOutAxis is out_axes and should be a constant value.";
  }
  auto axis_size_value = axis_size_abstract->BuildValue();
  if (axis_size_value == nullptr || !axis_size_value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The third input to VmapMatchOutAxis is axis size and should be a constant unsigned int64 "
                      << " value.";
  }
  auto inputs = fg_->add_parameter();
  auto out_axis = fg_->add_parameter();
  auto axis_size = fg_->add_parameter();

  auto inputs_abstract_sequence = dyn_cast<abstract::AbstractSequence>(inputs_abstract);
  MS_EXCEPTION_IF_NULL(inputs_abstract_sequence);
  auto inputs_abstract_elements = inputs_abstract_sequence->elements();
  auto inputs_abstract_elements_size = inputs_abstract_elements.size();
  if (inputs_abstract_elements_size == 0) {
    MS_LOG(EXCEPTION) << "The input to VmapMatchOutAxis is empty";
  }
  auto inputs_abstract_elements_begin = inputs_abstract_elements[0];
  auto inputs_abstract_elements_end = inputs_abstract_elements[inputs_abstract_elements_size - 1];
  CNodePtr out_cnode = nullptr;
  constexpr size_t kInputAbstractElementsSize = 2;
  if (inputs_abstract_elements_begin->isa<abstract::AbstractTuple>() &&
      inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
    // All elements in inputs are tuple. The format of input is ((x, x_axis), (y, y_axis), (z, z_axis)).
    out_cnode =
      GenerateFuncGraphInnerAllTuple(inputs, out_axis, axis_size, inputs_abstract_elements, out_axes_abstract);
  } else if (inputs_abstract_elements_begin->isa<abstract::AbstractTuple>() &&
             !inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
    // The last element of input is axis. The format is ((x, y), None).
    if (inputs_abstract_elements_size != kInputAbstractElementsSize) {
      MS_LOG(EXCEPTION) << "The length of elements should be 2 but got: " << inputs_abstract_elements_size << ".";
    }
    out_cnode = GenerateFuncGraphInnerBroadcastAxis(inputs, out_axis, axis_size, inputs_abstract_elements_begin);
  } else {
    // Single tuple element. (x, None)
    if (inputs_abstract_elements_size != kInputAbstractElementsSize) {
      MS_LOG(EXCEPTION) << "The length of elements should be 2 but got: " << inputs_abstract_elements_size << ".";
    }
    out_cnode = GenerateFuncGraphInnerSingleElement(inputs, out_axis, axis_size, inputs_abstract_elements_end);
  }
  fg_->set_output(out_cnode);
  return fg_;
}

FuncGraphPtr VmapGeneralPreprocess::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  auto prim = fg->add_parameter();
  auto args_size = args_abs_list.size();
  if (args_size <= 1) {
    MS_LOG(EXCEPTION) << "The length of input to VmapGeneralPreprocess must be greater than 1";
  }
  int64_t inputs_size = SizeToLong(args_size - 1);
  int64_t tuple_elements_num = 0;
  uint32_t offset = 1;
  auto get_tuple_elements = [args_size, &tuple_elements_num, &inputs_size,
                             &offset](const AbstractBasePtrList &args_abs_list) -> AbstractBasePtrList {
    auto arg = args_abs_list[1];
    if (!arg->isa<abstract::AbstractSequence>()) {
      MS_LOG(EXCEPTION) << "The second input to VmapGeneralPreprocess should be AbstractSequence but got: "
                        << arg->ToString() << ".";
    }
    auto arg_seq = arg->cast<abstract::AbstractSequencePtr>();
    const auto &arg_tuple_elements = arg_seq->elements();
    if (arg_tuple_elements.back()->isa<abstract::AbstractTuple>()) {
      // Operators with indefinite inputs length, such as `AddN`, whose inputs is wrapped
      // into a tuple. We need to process the internal elements separately and then re-wrap
      // them into tuple. Handle case such as args:(((A, 0), (B, 1), (C, None)), ...). Which
      // different from the case with single input parameter ((A, 0),).
      //
      // Tuple case:
      // 1. Only one tuple input: (((A, 0), (B, 1), (C, None)),)
      // 2. A tuple input and some normal inputs: (((A, 0), (B, 1), (C, None)), (a, 2), (b, 3))
      tuple_elements_num = arg_tuple_elements.size();
      inputs_size = tuple_elements_num + inputs_size - 1;
      offset = 0;
      AbstractBasePtrList unfold_args_abs_list(arg_tuple_elements.begin(), arg_tuple_elements.end());
      constexpr size_t unfold_index = 2;
      (void)unfold_args_abs_list.insert(unfold_args_abs_list.end(), args_abs_list.begin() + unfold_index,
                                        args_abs_list.end());  // the maybe left inputs.
      return unfold_args_abs_list;
    }
    return args_abs_list;
  };
  auto unfold_elements = get_tuple_elements(args_abs_list);
  bool is_all_none = true;
  constexpr size_t kCurTupleSize = 2;
  for (int64_t i = 0; i < inputs_size; ++i) {
    auto cur_arg = unfold_elements[i + offset];
    if (!cur_arg->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "The " << i + offset
                        << "th input to VmapGeneralPreprocess should be AbstractTuple but got: " << cur_arg->ToString()
                        << ".";
    }
    auto cur_arg_tuple = cur_arg->cast<abstract::AbstractTuplePtr>();
    auto cur_arg_tuple_elements = cur_arg_tuple->elements();
    if (cur_arg_tuple_elements.size() != kCurTupleSize) {
      MS_LOG(EXCEPTION) << "The " << i + offset << "th input to VmapGeneralPreprocess should be a tuple with two "
                        << "elements but got " << cur_arg_tuple_elements.size() << " elements.";
    }
    if (!cur_arg_tuple_elements[kDimIndex]->isa<abstract::AbstractNone>()) {
      MS_LOG(INFO) << "The " << i + offset << "th input to VmapGeneralPreprocess has not None dim value.";
      is_all_none = false;
      break;
    }
  }

  std::vector<AnfNodePtr> output_cnode_inputs;
  (void)output_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  if (!is_all_none) {
    for (size_t i = 1; i < args_size; ++i) {
      (void)fg->add_parameter();
    }
    auto output_cnode =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), NewValueNode(false), NewValueNode(kNone)});
    fg->set_output(output_cnode);
  } else {
    GenerateFuncGraphAllNone(fg, prim, inputs_size, tuple_elements_num, true);
  }
  return fg;
}

/// \brief ConstructMapInput.
///
/// \param[in] unfold_elements_abstract Unfold elements abstract, such as ((A, 0), (B, 0), (C, None)).
/// \param[in] args_size The size of elements.
/// \param[in] tuple_elements_num The elements-size for first tuple input.
/// \return A vector of AnfNodePtrList, the size is equal to vmap dim size.
CNodeInpusList VmapGeneralRule::ConstructMapInput(const InputsAbstractList &unfold_elements_abstract, int64_t args_size,
                                                  int64_t tuple_elements_num) {
  AnfNodePtr single_input = nullptr;
  if (tuple_elements_num != 0) {
    single_input = fg_->add_parameter();
  }

  CNodeInpusList map_inputs(axis_size_);
  for (int64_t i = 0; i < args_size; ++i) {
    AnfNodePtr cur_arg_node = nullptr;
    if (i < tuple_elements_num) {
      cur_arg_node = fg_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), single_input, NewValueNode(i)});
    } else {
      cur_arg_node = fg_->add_parameter();
    }
    auto unfold_element_abstract = unfold_elements_abstract[i];
    auto val_abstract = unfold_element_abstract[kValIndex];
    auto dim_abstract = unfold_element_abstract[kDimIndex];
    AnfNodePtr val_cnode =
      fg_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), cur_arg_node, NewValueNode(kValIndex)});

    if (dim_abstract->isa<abstract::AbstractNone>()) {
      for (int64_t m = 0; m < axis_size_; ++m) {
        map_inputs[m].push_back(val_cnode);
      }
    } else {
      if (!val_abstract->isa<abstract::AbstractTensor>()) {
        MS_LOG(EXCEPTION) << "A variable of type other than `Tensor` is accepted, but the source axis is not `None`";
      }
      AnfNodePtr dim_cnode =
        fg_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), cur_arg_node, NewValueNode(kDimIndex)});
      const py::function unstack_fn = python_adapter::GetPyFn(kVmapFunctionModelName, "vmap_unstack");
      auto unstack_fg_ = parse::ParsePythonCode(unstack_fn);
      MS_EXCEPTION_IF_NULL(unstack_fg_);
      auto out_cnode = fg_->NewCNodeInOrder({NewValueNode(unstack_fg_), dim_cnode, val_cnode});
      for (int64_t m = 0; m < axis_size_; ++m) {
        auto out_element_cnode =
          fg_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), out_cnode, NewValueNode(m)});
        map_inputs[m].push_back(out_element_cnode);
      }
    }
  }
  return map_inputs;
}

// When the primitive does not registered the relevant specific VmapRule, it attempts to get
// this the general rule. The general rule is combining loop and stack operators to simulate
// the behavior of Vmap. Noted that, general rules does not guarantee the correctness of
// execution results.
// Currently, only the following types of primitives are supported:
// 1、 Most calculation operations, whose inputs are tensors, scalars or both of them.
// (If all elements in a tuple are scalars, it is also considered scalar.)
// 2、 Operators with indefinite inputs length, such as `AddN`, whose inputs is wrapped into a tuple.
// 3、 Operators with indefinite inputs length, whose first inputs is wrapped into a tuple.
// In other words, we do not support any tuple wrapped variables except for the special cases
//   listed above.
FuncGraphPtr VmapGeneralRule::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  fg_ = std::make_shared<FuncGraph>();
  int64_t args_size = static_cast<int64_t>(args_abs_list.size());
  int64_t tuple_elements_num = 0;
  auto get_tuple_elements = [&args_size,
                             &tuple_elements_num](const AbstractBasePtrList &args_abs_list) -> AbstractBasePtrList {
    auto arg = args_abs_list[0];
    if (!arg->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "The first input to VmapGeneralPreprocess should be AbstractTuple but got: "
                        << arg->ToString() << ".";
    }
    auto arg_tuple = arg->cast<abstract::AbstractTuplePtr>();
    const auto &arg_tuple_elements = arg_tuple->elements();
    if (arg_tuple_elements.back()->isa<abstract::AbstractTuple>()) {
      // Operators with indefinite inputs length, such as `AddN`, whose inputs is wrapped
      // into a tuple. We need to process the internal elements separately and then re-wrap
      // them into tuple. Handle case such as args:(((A, 0), (B, 1), (C, None)), ...). Which
      // different from the case with single input parameter ((A, 0),).
      //
      // Tuple case:
      // 1. Only one tuple input: (((A, 0), (B, 1), (C, None)),)
      // 2. A tuple input and some normal inputs: (((A, 0), (B, 1), (C, None)), (a, 2), (b, 3))
      tuple_elements_num = arg_tuple_elements.size();
      args_size = tuple_elements_num + args_size - 1;
      AbstractBasePtrList unfold_args_abs_list(arg_tuple_elements.begin(), arg_tuple_elements.end());
      (void)unfold_args_abs_list.insert(unfold_args_abs_list.end(), args_abs_list.begin() + 1,
                                        args_abs_list.end());  // the maybe left inputs.
      return unfold_args_abs_list;
    }

    return args_abs_list;
  };
  auto unfold_elements = get_tuple_elements(
    args_abs_list);  // ((A, 0), (B, 1), ...), if tuple is the first input, its elements will be unfold.

  bool is_all_none = true;
  constexpr size_t kCurTupleSize = 2;
  InputsAbstractList unfold_elements_abstract(args_size);
  for (int64_t i = 0; i < args_size; ++i) {
    auto cur_arg = unfold_elements[i];
    if (!cur_arg->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "The " << i
                        << "th input to VmapGeneralPreprocess should be AbstractTuple but got: " << cur_arg->ToString()
                        << ".";
    }
    auto cur_arg_tuple = cur_arg->cast<abstract::AbstractTuplePtr>();
    auto cur_arg_tuple_elements = cur_arg_tuple->elements();
    if (cur_arg_tuple_elements.size() != kCurTupleSize) {
      MS_LOG(EXCEPTION) << "The " << i << "th input to VmapGeneralPreprocess should be a tuple with two "
                        << "elements but got " << cur_arg_tuple_elements.size() << " elements.";
    }
    auto dim_abstract = cur_arg_tuple_elements[kDimIndex];
    if (is_all_none && !dim_abstract->isa<abstract::AbstractNone>()) {
      MS_LOG(INFO) << "The " << i << "th input to VmapGeneralPreprocess has not None dim value.";
      is_all_none = false;
    }
    auto val_abstract = cur_arg_tuple_elements[kValIndex];
    std::vector<abstract::AbstractBasePtr> element_abstract = {val_abstract, dim_abstract};
    unfold_elements_abstract[i] = element_abstract;
  }

  if (is_all_none) {
    GenerateFuncGraphAllNone(fg_, NewValueNode(prim_), args_size, tuple_elements_num, false);
    return fg_;
  }

  CNodeInpusList map_inputs = ConstructMapInput(unfold_elements_abstract, args_size, tuple_elements_num);  //

  std::vector<AnfNodePtr> output_cnode_inputs;
  (void)output_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (auto map_input : map_inputs) {
    std::vector<AnfNodePtr> output_element_cnode_inputs;
    if (tuple_elements_num != 0) {
      std::vector<AnfNodePtr> tuple_cnode_inputs{NewValueNode(prim::kPrimMakeTuple)};
      (void)tuple_cnode_inputs.insert(tuple_cnode_inputs.cend(), map_input.cbegin(),
                                      map_input.cbegin() + tuple_elements_num);
      auto tuple_cnode = fg_->NewCNodeInOrder(tuple_cnode_inputs);
      output_element_cnode_inputs.push_back(NewValueNode(prim_));
      output_element_cnode_inputs.push_back(tuple_cnode);
      (void)output_element_cnode_inputs.insert(output_element_cnode_inputs.end(),
                                               map_input.cbegin() + tuple_elements_num, map_input.cend());
    } else {
      output_element_cnode_inputs.push_back(NewValueNode(prim_));
      (void)output_element_cnode_inputs.insert(output_element_cnode_inputs.cend(), map_input.cbegin(),
                                               map_input.cend());
    }
    auto output_element_cnode = fg_->NewCNodeInOrder(output_element_cnode_inputs);
    (void)output_cnode_inputs.emplace_back(output_element_cnode);
  }
  auto output_cnode = fg_->NewCNodeInOrder(output_cnode_inputs);
  const py::function vmap_general_output_process_fn =
    python_adapter::GetPyFn(kVmapFunctionModelName, "vmap_general_output_process");
  auto vmap_general_output_process_fg_ = parse::ParsePythonCode(vmap_general_output_process_fn);
  MS_EXCEPTION_IF_NULL(vmap_general_output_process_fg_);
  auto vmap_general_output = fg_->NewCNodeInOrder({NewValueNode(vmap_general_output_process_fg_), output_cnode});
  fg_->set_output(vmap_general_output);
  return fg_;
}
}  // namespace prim
}  // namespace mindspore
