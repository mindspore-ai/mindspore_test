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
#include "frontend/parallel/ops_info/triu_info.h"

namespace mindspore {
namespace parallel {
int64_t TriuInfo::GetDiag() {
  const auto &input_shape = inputs_shape_.at(0);
  auto row = *(input_shape.rbegin() + 1);
  auto col = *(input_shape.rbegin());
  auto stra = strategy();
  Strategies strategies = stra->GetInputDim();
  auto stra_value = strategies.at(0);
  auto c = *(stra_value.rbegin() + 1);
  auto d = *(stra_value.rbegin());
  int64_t rank = g_device_manager->rank_index_in_stage();
  auto t = row / c;
  auto u = col / d;
  // represent position in the row
  int64_t m = 0;
  // represent position in the col
  int64_t n = 0;
  if (repeated_calc_num_ > 1) {
    // repeated calc
    auto h = *(dev_matrix_shape().rbegin());
    m = (rank / d / h % c) * t;
    n = (rank / h % d) * u;
  } else {
    m = (rank / d % c) * t;
    n = (rank % d) * u;
  }
  // Numbers to be reserved in the first row of the fragment.
  auto z = m - n + diagonal_;
  // clip
  z = std::max(std::min(z, u), -t + 1);
  return z;
}

void TriuInfo::ReplaceNodeInputOrAttrs() {
  for (auto &node : cnodes_) {
    auto new_diag = GetDiag();
    MS_LOG(INFO) << name_ << ": the new diag is " << new_diag;
    ValuePtr diagonal = MakeValue(new_diag);
    AnfNodePtr val = NewValueNode(diagonal);
    node->set_input(kIndex2, val);
  }
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
