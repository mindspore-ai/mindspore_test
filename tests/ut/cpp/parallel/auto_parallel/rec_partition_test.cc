/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include "frontend/parallel/auto_parallel/stage_compute.h"
#include <memory>
#include "ir/value.h"

namespace mindspore {
namespace parallel {
#define ARRAY_A 3000  // also 'I' :height of the first input tensor
#define ARRAY_B 1000  // also 'K' :used by both input tensor
#define ARRAY_C 4000  // also 'J' :width of the first input tensor

class TestPartition : public UT::Common {
 public:
  void Create(std::shared_ptr<Graph> graph, int node_num, std::vector<int64_t> edge_head,
              std::vector<int64_t> edge_tail);
  void InitEdge(std::shared_ptr<Graph> graph, int vHead, int vTail);
  void InitNode(std::shared_ptr<Graph> graph, int num_node);
  TensorParam MakeTensor(int n, int c, int h, int w);
  std::shared_ptr<Graph> MakeMatMulData(int numNode);
};

// Local function to create test input graph with nodes
void TestPartition::Create(std::shared_ptr<Graph> graph, int node_num, std::vector<int64_t> edge_head,
                           std::vector<int64_t> edge_tail) {
  TestPartition::InitNode(graph, node_num);
  unsigned int edge_num = edge_head.size();
  if (edge_num != edge_tail.size()) {
    exit(1);
  };

  for (unsigned int i = 0; i < edge_num; i++) {
    TestPartition::InitEdge(graph, edge_head[i], edge_tail[i]);
  };
}

// Local function for Create() to crate Node
void TestPartition::InitNode(std::shared_ptr<Graph> graph, int num_node) {
  Graph::NodeType NewNode;
  for (int i = 0; i < num_node; i++) {
    graph->nodes.push_back(NewNode);
    std::stringstream ss;
    ss << 'N' << i;
    graph->nodes[i].name = ss.str();
    graph->nodes[i].info = kConstant;
  };
}

// Local function for Create() to crate Edge
void TestPartition::InitEdge(std::shared_ptr<Graph> graph, int vHead, int vTail) {
  NodeDep tail;
  tail.idx = vTail;
  graph->nodes[vHead].node_out.push_back(tail);
  graph->nodes[vTail].node_in.push_back(vHead);
}

// Local function for Create() to crate Tensor
TensorParam TestPartition::MakeTensor(int n, int c, int h, int w) {
  TensorParam tp;
  tp.tensor_shape.shape_n = n;
  tp.tensor_shape.shape_c = c;
  tp.tensor_shape.shape_h = h;
  tp.tensor_shape.shape_w = w;
  return std::move(tp);
};

// Local function for Create() to create MatMul Operator
// @numNode include Tensor and Operator, for example 4(1 Input Tensor, 1 Input Tensor, 1 Operator, 1 Output Tensor)
std::shared_ptr<Graph> TestPartition::MakeMatMulData(int numNode) {
  // Build Edges
  int edgeNum = 0;
  constexpr int INTERVAL = 2;
  if (numNode % INTERVAL == 0 && numNode != 0) {
    edgeNum = numNode - INTERVAL;
  } else if (numNode % INTERVAL == 1) {
    edgeNum = numNode - 1;
  } else {
    edgeNum = 0;
  };

  std::vector<int64_t> edgeHead(edgeNum);  // int edgeHead[8] = {0,2,4,6,1,3,5,7};
  std::vector<int64_t> edgeTail(edgeNum);  // int edgeTail[8] = {2,4,6,8,2,4,6,8};
  std::vector<std::string> node_param_name_vec = {".projection.weight", ".mapping.weight", ".attention.dense2.weight",
     ".attention_norm.weight"};

  for (int i = 0; i < edgeNum; i++) {
    edgeHead[i] = i;
    if (i % INTERVAL == 0) {
      edgeTail[i] = i + INTERVAL;
    } else {
      edgeTail[i] = i + 1;
    };
  };

  // Create graph
  std::shared_ptr<Graph> graph(new Graph);
  graph->dyn_shape_tmp_fix = true; //TEST
  TestPartition::Create(graph, numNode, edgeHead, edgeTail);
  int k = 0;

  // Add Node information.
  for (int i = 0; i < numNode; i++) {
    if (0 == i) {
      graph->nodes[i].info               = InfoType::kConstant;
      graph->nodes[i].tensor_parm        = MakeTensor(1, 1, ARRAY_A, ARRAY_B);

    } else if (0 == i % 4) {
      graph->nodes[i].info               = InfoType::kApplication;
      graph->nodes[i].apply.op_type      = OperatorType::kRecMatMul;
      graph->nodes[i].param_name         = node_param_name_vec[k++];
      graph->nodes[i].apply.arguments[0] = MakeTensor(1, 1, ARRAY_A, ARRAY_C);
      graph->nodes[i].apply.arguments[1] = MakeTensor(1, 1, ARRAY_C, ARRAY_B);
      graph->nodes[i].tensor_parm        = MakeTensor(1, 1, ARRAY_A, ARRAY_B);

    } else if (1 == i % 4) {
      graph->nodes[i].info               = InfoType::kConstant;
      graph->nodes[i].tensor_parm        = MakeTensor(1, 1, ARRAY_B, ARRAY_C);

    } else if (2 == i % 4) {
      graph->nodes[i].info               = InfoType::kApplication;
      graph->nodes[i].apply.op_type      = OperatorType::kRecMatMul;
      graph->nodes[i].param_name         = node_param_name_vec[k++];
      graph->nodes[i].apply.arguments[0] = MakeTensor(1, 1, ARRAY_A, ARRAY_B);
      graph->nodes[i].apply.arguments[1] = MakeTensor(1, 1, ARRAY_B, ARRAY_C);
      graph->nodes[i].tensor_parm        = MakeTensor(1, 1, ARRAY_A, ARRAY_C);

    } else if (3 == i % 4) {
      graph->nodes[i].info               = InfoType::kConstant;
      graph->nodes[i].tensor_parm        = MakeTensor(1, 1, ARRAY_C, ARRAY_B);

    };
  };
  return graph;
};

TEST_F(TestPartition, test_GetWeights) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);

  double wop1 = GetWeights(graph->nodes[2]);
  double wop2 = GetWeights(graph->nodes[4]);
  double wop3 = GetWeights(graph->nodes[6]);
  double wop4 = GetWeights(graph->nodes[8]);
  ASSERT_GE(wop1, wop2);
  ASSERT_LE(wop2, wop3);
  ASSERT_GE(wop3, wop4);
}

/// Feature: test GetWeights with different OperatorType
/// Description:
/// Expectation: success
TEST_F(TestPartition, test_GetWeights2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(3);

  graph->nodes[2].apply.op_type = OperatorType::kRecConvolution;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecPooling;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecElmWiseOp;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecReLU;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecReshape;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecBiasAdd;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecLog;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecSoftmax;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecStandAlone;
  GetWeights(graph->nodes[2]);

  graph->nodes[2].apply.op_type = OperatorType::kRecBatchMatMul;
  GetWeights(graph->nodes[2]);
}

TEST_F(TestPartition, test_SortByWeight) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  std::vector<size_t> result = SortByWeight(graph);
  ASSERT_GE(result.at(0), result.at(1));
  ASSERT_GE(result.at(1), result.at(2));
  ASSERT_GE(result.at(2), result.at(3));
}

TEST_F(TestPartition, test_SortByWeight2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(5);
  std::vector<size_t> result = SortByWeight(graph);
  ASSERT_GE(result.at(0), result.at(1));
}

TEST_F(TestPartition, test_PartitionNode) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  // node 2 is the first kRecMatMul Operator
  Graph::NodeType node2 = graph->nodes[2];
  std::vector<std::pair<std::string, StrategyRec>> nameToStrategy;
  bool isTraining = true;
  StrategyRec str = PartitionNode(node2, nameToStrategy, graph, isTraining, 0);
  ASSERT_EQ(str.outputTensor.str_h, 1);
  ASSERT_EQ(str.outputTensor.str_w, 1);
}

/// Feature: test PartitionNode with nodes of type kRecBatchMatMul
/// Description:
/// Expectation: success
TEST_F(TestPartition, test_PartitionNode2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);

  graph->nodes[2].apply.op_type = OperatorType::kRecBatchMatMul;
  graph->nodes[4].apply.op_type = OperatorType::kRecBatchMatMul;
  graph->nodes[6].apply.op_type = OperatorType::kRecBatchMatMul;
  graph->nodes[8].apply.op_type = OperatorType::kRecBatchMatMul;

  Graph::NodeType node = graph->nodes[2];
  std::vector<std::pair<std::string, StrategyRec>> nameToStrategy;
  bool isTraining = true;
  StrategyRec str = PartitionNode(node, nameToStrategy, graph, isTraining, 0);
  ASSERT_EQ(str.outputTensor.str_h, 1);
  ASSERT_EQ(str.outputTensor.str_w, 1);

  node = graph->nodes[4];
  str = PartitionNode(node, nameToStrategy, graph, isTraining, 0);
  ASSERT_EQ(str.outputTensor.str_h, 1);
  ASSERT_EQ(str.outputTensor.str_w, 0.5);

  node = graph->nodes[6];
  str = PartitionNode(node, nameToStrategy, graph, isTraining, 0);
  ASSERT_EQ(str.outputTensor.str_h, 1);
  ASSERT_EQ(str.outputTensor.str_w, 0.5);
}

TEST_F(TestPartition, test_PartitionForAllDevices) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  ASSERT_EQ(PartitionForAllDevices(1024, device_memory, graph, isTraining, nullptr), SUCCESS);
}

TEST_F(TestPartition, test_PartitionForAllDevices2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  ASSERT_EQ(PartitionForAllDevices(2, device_memory, graph, isTraining, nullptr), SUCCESS);
}

// Negative case: partition on 0 device
TEST_F(TestPartition, test_PartitionForAllDevices0) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  // Throw Exception "Number of devices can't be 0"
  EXPECT_ANY_THROW(PartitionForAllDevices(0, device_memory, graph, isTraining, nullptr));
}

TEST_F(TestPartition, test_ApplyStrToTensor) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  std::vector<std::pair<std::string, StrategyRec>> nameToStrategy;
  bool isTraining = true;
  graph->nodes[4].apply.str = PartitionNode(graph->nodes[4], nameToStrategy, graph, isTraining, 0);
  auto h_str = graph->nodes[4].apply.str.outputTensor.str_h;
  auto w_str = graph->nodes[4].apply.str.outputTensor.str_w;

  Graph::NodeType n_node = ApplyStrToTensor(graph->nodes[4]);
  auto h_node = n_node.tensor_parm.tensor_str.str_h;
  auto w_node = n_node.tensor_parm.tensor_str.str_w;
  ASSERT_EQ(h_str, h_node);
  ASSERT_EQ(w_str, w_node);
}

/// Feature: test GetDPAndMP.
/// Description:
/// Expectation: success
TEST_F(TestPartition, test_get_dp_mp) {
  size_t dp, mp;
  bool isTraining = true;
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  PartitionForAllDevices(8, device_memory, graph, isTraining, nullptr);
  std::tie(dp, mp) = GetDPAndMP(graph, 1);
  ASSERT_GT(dp, 0);
  ASSERT_GT(mp, 0);
  ASSERT_LE(dp, GetNumDevices());
  ASSERT_LE(mp, GetNumDevices());
}

}  // namespace parallel
}  // namespace mindspore
