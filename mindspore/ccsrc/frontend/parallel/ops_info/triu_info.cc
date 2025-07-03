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

#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"

#include "frontend/parallel/ops_info/triu_info.h"

namespace mindspore {
namespace parallel {

std::tuple<int64_t, int64_t, int64_t, int64_t> TriuInfo::GetSliceInfo() {
  auto stra = strategy();
  int64_t c = 0;
  int64_t d = 0;
  if (stra == nullptr) {
    auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
    auto layout_value = input_layout0.device_arrangement_origin().array();
    c = *(layout_value.rbegin() + 1);
    d = *(layout_value.rbegin());
  } else {
    auto stra_value = stra->GetInputDim()[kIndex0];
    c = *(stra_value.rbegin() + 1);
    d = *(stra_value.rbegin());
  }
  // represent position in the row
  int64_t m = 0;
  // represent position in the col
  int64_t n = 0;
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();
  if (repeated_calc_num_ > 1) {
    // repeated calc
    auto h = *(dev_matrix_shape().rbegin());
    m = (rank / d / h % c);
    n = (rank / h % d);
  } else {
    m = (rank / d % c);
    n = (rank % d);
  }
  return {m, n, c, d};
}

int64_t TriuInfo::GetDiag() {
  auto [m, n, c, d] = GetSliceInfo();

  const auto &input_shape = inputs_shape_.at(0);
  auto row = *(input_shape.rbegin() + 1);
  auto col = *(input_shape.rbegin());
  auto t = row / c;
  auto u = col / d;
  // Numbers to be reserved in the first row of the fragment.
  auto z = m * t - n * u + diagonal_;
  // clip
  z = std::max(std::min(z, u), -t + 1);
  return z;
}

void TriuInfo::ReplaceNodeInputOrAttrs() {
  const auto &input_shape = inputs_shape_.at(0);
  auto row = *(input_shape.rbegin() + 1);
  auto col = *(input_shape.rbegin());
  // The new value of "diagonal" depends only on the last two dimensions of the input shape.
  // Dynamic shape case:
  if (row < 0 || col < 0) {
    MS_LOG(INFO) << name_ << ": ReplaceNodeInputOrAttrs works for static shape only, bug got row: " << row
                 << ", and col: " << col;
    return;
  }
  // Static shape case:
  for (auto &node : cnodes_) {
    auto new_diag = GetDiag();
    MS_LOG(INFO) << name_ << ": new_diag is " << new_diag;
    ValuePtr diagonal = MakeValue(new_diag);
    AnfNodePtr val = NewValueNode(diagonal);
    node->set_input(kIndex2, val);
  }
}

ReplaceGraphPtr TriuInfo::replace_graph(const CNodePtr &cnode) {
  const auto &input_shape = inputs_shape_.at(0);
  auto row = *(input_shape.rbegin() + 1);
  auto col = *(input_shape.rbegin());
  // The new value of "diagonal" depends only on the last two dimensions of the input shape.
  // Static shape case:
  if (row >= 0 && col >= 0) {
    MS_LOG(INFO) << name_ << ": replace_graph works for dynamic shape only, bug got row: " << row
                 << ", and col: " << col;
    return replace_graph_;
  }
  // Dynamic shape case:
  if (ReplaceGraphForDynamicShape(cnode) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": ReplaceGraphForDynamicShape failed.";
  }
  return replace_graph_;
}

Status TriuInfo::ReplaceGraphForDynamicShape(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed.";
    return FAILED;
  }

  auto [m, n, c, d] = GetSliceInfo();
  std::ignore = c;
  std::ignore = d;

  auto inputs_shape = gen_g.PushBack({gen_g.NewOpInst(SHAPE_OP), gen_g.virtual_input_node()});
  auto row = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), inputs_shape, CreatInt64Imm(-2)});
  auto col = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), inputs_shape, CreatInt64Imm(-1)});

  // t = row and u = col after shape op executed.
  auto t_mul_m = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL), row, CreatInt64Imm(m)});
  auto u_mul_n = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL), col, CreatInt64Imm(n)});
  auto offset = gen_g.PushBack({gen_g.NewOpInst(SCALAR_SUB), t_mul_m, u_mul_n});
  auto z = gen_g.PushBack({gen_g.NewOpInst(SCALAR_ADD), offset, gen_g.virtual_input_node()});
  // clip
  // z = std::max(std::min(z, u), -t + 1);
  auto min_z_u = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MIN), z, col});
  auto neg_t_plus_1 = gen_g.PushBack({gen_g.NewOpInst(SCALAR_SUB), CreatInt64Imm(1), row});
  auto new_diag = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MAX), min_z_u, neg_t_plus_1});
  auto triu = gen_g.PushBack({gen_g.NewOpInst(TRIU), gen_g.virtual_input_node(), new_diag});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes = {std::make_pair(inputs_shape, 1), std::make_pair(z, 2),
                                                              std::make_pair(triu, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, triu));
  return SUCCESS;
}

Status TriuInfo::GetAttrs() {
  auto diagonal_value = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, DIAGONAL);
  if (!diagonal_value.has_value()) {
    return FAILED;
  }
  diagonal_ = diagonal_value.value();
  return SUCCESS;
}

REGISTER(TriuInfo);
}  // namespace parallel
}  // namespace mindspore
