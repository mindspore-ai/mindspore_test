/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_
#define PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <climits>

#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "ir/anf.h"

namespace mindspore {
namespace parallel {
enum OperatorType {
  kRecUnknownType,
  kRecMatMul,
  kRecConvolution,
  kRecPooling,
  kRecElmWiseOp,
  kRecReLU,
  kRecBatchNorm,
  kRecLayerNorm,
  kRecReshape,
  kRecBiasAdd,
  kRecSoftmax,
  kRecSparseSoftmaxCrossEntropyWithLogits,
  kRecSoftmaxCrossEntropyWithLogits,
  kRecOneHot,
  kRecLog,
  kRecExp,
  kRecAdd,
  kRecSub,
  kRecMul,
  kRecDiv,
  kRecSqueeze,
  kRecCast,
  kRecReduce,
  kRecPReLU,
  kRecGatherV2,
  kRecExpandDims,
  kRecStridedSlice,
  kRecArgWithValue,
  kRecUnsortedSegmentOp,
  kRecBatchMatMul,
  kRecFlatten,
  kRecCum,
  kRecStandAlone,
  kRecBatchParallel,
  kRecPadV3,
  kRecVirtual,
  kFlashAttentionScore,
  kRecRmsNorm
};

enum InfoType { kApplication, kConstant };

struct OperatorRec {
  OperatorType op_type;
  TensorParam arguments[MAX_INPUT_NUM];
  StrategyRec str;
  std::vector<StrategyRec> strs;
};

struct NodeDep {
  size_t idx;
  std::vector<int64_t> transpose_mapping;
  std::vector<std::vector<int64_t>> reshape_mapping;
};

// Define simplified dataflow Graph for partitioning
class Graph {
 public:
  struct NodeType {
    std::string name;
    // Nodes that point to this node
    std::vector<size_t> node_in;
    // Nodes that point from this node
    std::vector<NodeDep> node_out;
    // Nodes that point to this node via auxiliary edges
    std::vector<size_t> node_in_aux;
    // Input indices of the nodes that point to this node via auxliary edges
    std::vector<size_t> node_in_aux_idx;
    //  operation of transpose
    std::vector<int64_t> transpose_mapping;
    // operation of reshape
    std::vector<std::vector<int64_t>> reshape_mapping;
    // Node Type Info: Application or Constant. Defined in enum <InfoType> .
    InfoType info;
    // Operator info. Defined in struct <OperatorRec> .
    OperatorRec apply;
    // Tensor info. Defined in tensor.h struct <TensorParam> .
    TensorParam tensor_parm;

    std::string param_name;
    // yield to user-defined strategy
    bool interfered_sapp = false;
  };

  bool dyn_shape_tmp_fix = false;

  int64_t micro_batch_size = 1;
  // Nodes of the graph. Public.
  std::vector<Graph::NodeType> nodes;
};

inline std::vector<int64_t> TransposeCombine(const std::vector<int64_t> &tranpose_mapping,
                                             const std::vector<int64_t> &node_out_tranpose_mapping) {
  std::vector<int64_t> updated;

  if (tranpose_mapping.empty()) {
    return node_out_tranpose_mapping;
  }

  if (node_out_tranpose_mapping.empty()) {
    return tranpose_mapping;
  }

  if (tranpose_mapping.size() != node_out_tranpose_mapping.size()) {
    MS_LOG(EXCEPTION) << "tranpose_mapping " << tranpose_mapping << " and node_out_tranpose_mapping "
                      << node_out_tranpose_mapping << " should share same size";
  }

  MS_LOG(INFO) << "tranpose_mapping " << tranpose_mapping << " and node_out_tranpose_mapping "
               << node_out_tranpose_mapping;
  updated.insert(updated.begin(), node_out_tranpose_mapping.size(), 0);
  for (size_t i = 0; i < node_out_tranpose_mapping.size(); i++) {
    updated[i] = tranpose_mapping[node_out_tranpose_mapping[i]];
  }
  MS_LOG(INFO) << "after operation updating, mapping is: " << updated;

  return updated;
}

inline std::vector<std::vector<int64_t>> ReshapeCombine(
  const std::vector<std::vector<int64_t>> &reshape_mapping,
  const std::vector<std::vector<int64_t>> &node_out_reshape_mapping) {
  if (reshape_mapping.empty()) {
    return node_out_reshape_mapping;
  }

  if (node_out_reshape_mapping.empty()) {
    return reshape_mapping;
  }

  std::vector<std::vector<int64_t>> updated;

  if (reshape_mapping.size() != node_out_reshape_mapping.size()) {
    MS_LOG(EXCEPTION) << "reshape_mapping " << reshape_mapping << " and node_out_reshape_mapping "
                      << node_out_reshape_mapping << " should share same size";
  }

  MS_LOG(INFO) << "reshape_mapping " << reshape_mapping << " and node_out_reshape_mapping " << node_out_reshape_mapping;
  for (size_t i = 0; i < node_out_reshape_mapping.size(); i++) {
    std::vector<int64_t> tmp;
    if (node_out_reshape_mapping[i].empty()) {
      tmp.push_back(i);
    }
    for (int64_t ii : node_out_reshape_mapping[i]) {
      if (ii == INT_MAX) continue;
      tmp.insert(tmp.end(), reshape_mapping[ii].begin(), reshape_mapping[ii].end());
    }
    std::set<int64_t> s(tmp.begin(), tmp.end());
    tmp.assign(s.begin(), s.end());

    updated.push_back(tmp);
  }
  MS_LOG(INFO) << "after operation updating, mapping is: " << updated;

  return updated;
}

}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_
