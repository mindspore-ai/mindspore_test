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

#include "frontend/parallel/ops_info/arange_info.h"

#include <utility>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status ArangeInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (split_num_ == 1) {
    MS_LOG(INFO) << name_ << ": split num is 1, no need to replace graph";
    return SUCCESS;
  }

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateGraph Init failed.";
    return FAILED;
  }

  MS_EXCEPTION_IF_ZERO("split_num_", split_num_);
  int64_t slice_output_size = output_size_ / split_num_;
  AnfNodePtr step = gen_g.virtual_input_node();
  InferSliceId();
  OperatorAttrs keep_alive_attr = {std::make_pair(KEEP_ALIVE, MakeValue(true))};
  // single_slice_offset = slice_output_size * step
  // start_offset = slice_id * single_slice_offset
  // new_start = start + start_offset
  // new_end = new_start + single_slice_offset
  // new_step = step
  auto single_slice_offset = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL), step, CreatInt64Imm(slice_output_size)});
  auto start_offset =
    gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL, keep_alive_attr), single_slice_offset, CreatInt64Imm(slice_id_)});
  auto new_start = gen_g.PushBack({gen_g.NewOpInst(SCALAR_ADD), gen_g.virtual_input_node(), start_offset});
  auto new_end = gen_g.PushBack({gen_g.NewOpInst(SCALAR_ADD), new_start, single_slice_offset});
  auto arange = gen_g.PushBack({gen_g.NewOpInst(ARANGE), new_start, new_end, step, dtype_});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes = {std::make_pair(single_slice_offset, 3),
                                                              std::make_pair(new_start, 1), std::make_pair(arange, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, arange));
  MS_LOG(INFO) << name_ << ": output_size_ " << output_size_ << ", split_num_ " << split_num_;
  return SUCCESS;
}

REGISTER(ArangeInfo);
}  // namespace parallel
}  // namespace mindspore
