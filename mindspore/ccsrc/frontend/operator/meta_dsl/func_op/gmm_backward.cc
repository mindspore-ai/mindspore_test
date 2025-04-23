/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_backward.h"

#include <string>
#include <utility>

#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_common_utils.h"

namespace mindspore::prim {
void CheckGmmBackwardInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const std::string &op_name = primitive->name();
  CheckNumOfSequenceTensor(input_args.at(kIndex1), kIndex1, op_name, "x");
  CheckNumOfSequenceTensor(input_args.at(kIndex2), kIndex1, op_name, "weight");
  const auto num = MakeValue<int64_t>(1);
  primitive->AddAttr("x_num", num);
  primitive->AddAttr("weight_num", num);
}

BeginFunction(GmmBackward, grad, x, weight, group_list, group_list_type) {
  auto MakeRangeFunc = [&](const NodePtr &tensors) {
    auto tensors_num = SequenceLen(tensors);
    auto num_list = Call(Prim(MakeRange), tensors_num);
    return num_list;
  };

  auto ForEachTranspose = [&](const NodePtr &tensors, int64_t num) {
    std::vector<NodePtr> new_tensors;
    for (int64_t i = 0; i < num; ++i) {
      auto tensor_i = GetItem(tensors, Value(i));
      auto tensor_i_t = Call(Prim(TransposeExtView), tensor_i, Value(-1), Value(-2));
      new_tensors.push_back(std::move(tensor_i_t));
    }
    return MakeTuple(new_tensors);
  };

  auto ForEachReShape = [&](const NodePtr &tensors, const NodePtr &target_tensors) {
    auto idxes = MakeRangeFunc(tensors);

    auto loop_func = [&](const NodePtr &input, const NodePtr &idx) {
      auto ori_tensors = GetItem(input, Value(0));
      auto tar_tensors = GetItem(input, Value(1));
      auto ori_tensor_i = GetItem(ori_tensors, idx);
      auto tar_tensor_i = GetItem(tar_tensors, idx);
      auto new_tensor_i = Reshape(ori_tensor_i, Shape(tar_tensor_i));
      Return(Tuple(input, new_tensor_i));
    };

    auto out = Scan(loop_func, Tuple(tensors, target_tensors), idxes);

    return ListToTuple(GetItem(out, Value(1)));
  };

  auto Gmm = [&](const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                 int64_t group_type_value) -> NodePtr {
    auto split_item = Value(3);
    auto group_type = Value(group_type_value);
    auto none = NewValueNode(kNone);
    return Call(Prim(GroupedMatmulV2), x, weight, none, none, none, none, none, group_list, split_item, group_type);
  };

  const auto &primitive = prim();
  auto x_num = GetValue<int64_t>(primitive->GetAttr("x_num"));
  auto w_num = GetValue<int64_t>(primitive->GetAttr("weight_num"));

  auto xt = ForEachTranspose(x, x_num);
  auto wt = ForEachTranspose(weight, w_num);

  auto dx = Gmm(grad, wt, group_list, 0);
  auto dw = Gmm(xt, grad, group_list, 2);
  auto dw_reshape = ForEachReShape(dw, weight);

  auto SequenceAddFunc = [&](const NodePtr &sequence0, const NodePtr &sequence1) {
    auto sequence0_len = SequenceLen(sequence0);

    auto multi_elements_branch = [&]() {
      auto sequence1_len = SequenceLen(sequence1);
      auto num_list = Call(Prim(MakeRange), ScalarAdd(sequence0_len, sequence1_len));

      auto loop_func = [&](const NodePtr &all_inputs, const NodePtr &idx) {
        auto sequence_left = GetItem(all_inputs, Value(0));
        auto sequence_left_len = GetItem(all_inputs, Value(2));
        auto sequence_right = GetItem(all_inputs, Value(1));

        auto get_item = [&]() {
          auto true_branch = [&]() { Return(GetItem(sequence_left, idx)); };
          auto false_branch = [&]() { Return(GetItem(sequence_right, ScalarSub(idx, sequence_left_len))); };
          auto condition = Less(idx, sequence_left_len);
          return If(condition, true_branch, false_branch, (sequence_left, sequence_right, sequence_left_len, idx));
        };
        auto input_i = get_item();

        Return(Tuple(all_inputs, input_i));
      };

      auto loop_out = Scan(loop_func, Tuple(sequence0, sequence1, sequence0_len), num_list);
      auto out = ListToTuple(GetItem(loop_out, Value(1)));
      Return(out);
    };

    auto one_element_branch = [&]() {
      auto out = Tuple(GetItem(sequence0, Value(0)), GetItem(sequence1, Value(0)));
      Return(out);
    };

    return If(Equal(sequence0_len, Value(1)), one_element_branch, multi_elements_branch,
              (sequence0, sequence1, sequence0_len));
  };

  Return(SequenceAddFunc(dx, dw_reshape));
}
EndFunction(GmmBackward)
}  // namespace mindspore::prim
