/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "frontend/expander/bprop/common_utils.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore::expander::bprop {
std::string GetStringFromNode(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value_ptr = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The value of node is not a string, node:" << node->ToString();
  }

  return GetValue<std::string>(value_ptr);
}

REG_BPROP_BUILDERS_BEGIN(GradCommOps)
REG_BPROP_BUILDER("AllReduce").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto op = GetValue<std::string>(ib->GetAttr("op"));
  auto dy = dout;
  if (op == "prod") {
    dy = ib->Mul(dout, out);
  }
  auto dx = ib->Emit(kAllReduceOpName, {dy},
                     {{"op", MakeValue("sum")},
                      {"group", ib->GetAttr("group")},
                      {"index", ib->GetAttr("index")},
                      {"fusion", ib->GetAttr("fusion")},
                      {"no_eliminate", ib->GetAttr("no_eliminate")}});
  dx->set_debug_info("grad" + dx->debug_info());
  if (op == "prod") {
    return {ib->RealDiv(dx, x)};
  } else if (op == "sum") {
    return {dx};
  } else {
    auto z = ib->Equal(x, out);
    z = ib->Cast(z, ib->GetDtype(dx));
    return {ib->Mul(dx, z)};
  }
});

REG_BPROP_BUILDER("NeighborExchange").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("NeighborExchange", {dout},
                     {{"send_rank_ids", ib->GetAttr("recv_rank_ids")},
                      {"recv_rank_ids", ib->GetAttr("send_rank_ids")},
                      {"recv_shapes", ib->GetAttr("send_shapes")},
                      {"send_shapes", ib->GetAttr("recv_shapes")},
                      {"recv_type", ib->GetAttr("recv_type")},
                      {"group", ib->GetAttr("group")}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
  return {dx};
});

REG_BPROP_BUILDER("AllGather").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kReduceScatterOpName, {dout},
                     {{"op", MakeValue("sum")},
                      {"rank_size", ib->GetAttr("rank_size")},
                      {"group", ib->GetAttr("group")},
                      {"fusion", ib->GetAttr("fusion")},
                      {"no_eliminate", MakeValue(true)}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
  auto rank_size = GetValue<int64_t>(ib->GetAttr("rank_size"));
  if (rank_size == 0) {
    MS_LOG(EXCEPTION) << "The 'rank_size' can not be zero, but got" << rank_size;
  }
  auto mean_flag = GetValue<bool>(ib->GetAttr("mean_flag"));
  if (mean_flag) {
    auto scale = ib->Tensor(1.0 / rank_size, kFloat32);
    dx = ib->Mul(dx, scale);
  }
  return {dx};
});

REG_BPROP_BUILDER("_MirrorOperator").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dev_num = GetValue<int64_t>(ib->GetAttr("dev_num"));
  bool mean_flag = GetValue<bool>(ib->GetAttr("mean_flag"));
  if (dev_num == 1) {
    return {dout};
  }
  DAttr attrs{{"op", MakeValue("sum")},
              {"group", ib->GetAttr("group")},
              {"fusion", ib->GetAttr("fusion")},
              {"no_eliminate", MakeValue(true)}};
  if (ib->GetAttr("parameter") != nullptr) {
    (void)attrs.emplace_back("parameter", ib->GetAttr("parameter"));
  }
  auto dx = ib->Emit(kAllReduceOpName, {dout}, attrs);
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad_mirror" + ins_name);
  if (mean_flag) {
    dx = ib->Mul(dx, ib->Tensor(1.0 / dev_num, ib->GetDtype(dx)));
  }
  return {dx};
});

REG_BPROP_BUILDER("InnerCommAllReduce").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto op = ib->GetInput(kIndex1);
  const auto &op_type = GetStringFromNode(op);
  auto group = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);

  auto dy = dout;
  if (op_type == "prod") {
    dy = ib->Mul(dout, out);
  }

  auto group_value = group->BuildValue();
  auto dx = ib->Emit(kAllReduceOpName, {dy},
                     {{"op", MakeValue("sum")},
                      {"group", group_value},
                      {"index", MakeValue<int64_t>(0)},
                      {"fusion", MakeValue<int64_t>(0)},
                      {"no_eliminate", MakeValue(true)}});
  dx->set_debug_info("grad" + dx->debug_info());
  if (op_type == "prod") {
    dx = ib->RealDiv(dx, x);
  } else if (op_type != "sum") {
    auto z = ib->Equal(x, out);
    z = ib->Cast(z, ib->GetDtype(dx));
    dx = ib->Mul(dx, z);
  }

  return {dx, ib->OutZeros(op), ib->OutZeros(group)};
});

REG_BPROP_BUILDER("InnerCommAllGather").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto rank_size = ib->GetInput(kIndex1);
  auto group = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);

  auto rank_size_value = rank_size->BuildValue();
  auto group_value = group->BuildValue();
  auto dx = ib->Emit(kReduceScatterOpName, {dout},
                     {{"op", MakeValue("sum")},
                      {"rank_size", rank_size_value},
                      {"group", group_value},
                      {"fusion", MakeValue<int64_t>(0)},
                      {"no_eliminate", MakeValue(true)}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);

  return {dx, ib->OutZeros(rank_size), ib->OutZeros(group)};
});

REG_BPROP_BUILDER("InnerCommReduceScatter").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto rank_size = ib->GetInput(kIndex1);
  auto op = ib->GetInput(kIndex2);
  auto op_type = GetStringFromNode(op);
  auto group = ib->GetInput(kIndex3);

  if (op_type == "sum") {
    MS_LOG(EXCEPTION) << "The reducescatter bprop only support ReduceOp.SUM until now.";
  }

  auto group_value = group->BuildValue();

  auto dout = ib->GetInput(kIndex5);
  auto dx = ib->Emit(kAllGatherOpName, {dout}, {{"group", group_value}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
  return {dx, ib->OutZeros(rank_size), ib->OutZeros(op), ib->OutZeros(group)};
});

REG_BPROP_BUILDER("InnerCommIrecv").SetUnusedInputs({i0, i3, i5, i6}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto tag = ib->GetInput(kIndex1);
  auto rank = ib->GetInput(kIndex2);
  auto shape = ib->GetInput(kIndex3);
  auto group = ib->GetInput(kIndex4);
  auto dtype_node = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex7);

  auto out_tensor = ib->Tensor(0.0, kFloat16);
  auto dtype = ib->GetDtype(input);

  auto send_out =
    ib->Emit(kSendOpName, {dout},
             {{"sr_tag", tag->BuildValue()}, {"dest_rank", rank->BuildValue()}, {"group", group->BuildValue()}});
  auto dx = ib->Depend(ib->Cast(out_tensor, dtype), send_out);

  return {
    dx, ib->OutZeros(tag), ib->OutZeros(rank), ib->OutZeros(shape), ib->OutZeros(group), ib->OutZeros(dtype_node)};
});

REG_BPROP_BUILDER("InnerCommIsend").SetUnusedInputs({i0, i4, i5}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto rank = ib->GetInput(kIndex1);
  auto group = ib->GetInput(kIndex2);
  auto tag = ib->GetInput(kIndex3);

  auto shape = ib->GetShape(input);
  auto dtype = ib->GetDtype(input);
  auto virtual_input = ib->Tensor(0.0, dtype);

  auto dx = ib->Emit(kReceiveOpName, {virtual_input},
                     {{"sr_tag", tag->BuildValue()},
                      {"src_rank", rank->BuildValue()},
                      {"shape", MakeValue(shape)},
                      {"dtype", dtype},
                      {"group", group->BuildValue()}});

  return {dx, ib->OutZeros(rank), ib->OutZeros(group), ib->OutZeros(tag)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
