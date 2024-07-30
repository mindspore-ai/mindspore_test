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

#include <common/common_test.h>
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {

//TEST MatMulCost *************************************************
class TestMatMulCost : public UT::Common {
 public:
  TestMatMulCost() {}
  void SetUp();
  void TearDown();
  MatMulCost mmcost_;
};

void TestMatMulCost::SetUp() {
  mmcost_ = MatMulCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestMatMulCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for MatMul
/// Description:
/// Expectation: success
TEST_F(TestMatMulCost, test_CostGeneration) {
  // Currently, the implementation of GetForwardCommCost() method
  // does not check the tensor layouts information, instead, it checks the
  // tensor shape and the slice shape.
  TensorLayout input0_layout, input1_layout, output0_layout;
  Shape input0_shape{200, 300}, input1_shape{300, 500}, output0_shape{200, 500};
  Shape input0_slice_shape{20, 50}, input1_slice_shape{50, 25}, output0_slice_shape{20, 25};
  TensorInfo input0(input0_layout, input0_shape, input0_slice_shape),
    input1(input1_layout, input1_shape, input1_slice_shape),
    output0(output0_layout, output0_shape, output0_slice_shape);

  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input0);
  inputs.push_back(input1);
  outputs.push_back(output0);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  mmcost_.set_is_parameter({false, false});
  mmcost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  mmcost_.GetComputationCost(inputs, outputs, 0);
  mmcost_.GetForwardCommCost(inputs, outputs, 0);
  mmcost_.GetBackwardCommCost(inputs, outputs, 0);
  mmcost_.GetForwardComputationCost(inputs, outputs, 0);
  mmcost_.GetForwardComputationCost(inputs, outputs, 0);
}

class TestActivationCost : public UT::Common {
 public:
  TestActivationCost() {}
  void SetUp();
  void TearDown();
  ActivationInfoCost ac_cost_;
};

void TestActivationCost::SetUp() {
  ac_cost_ = ActivationInfoCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestActivationCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for Activation 
/// Description:
/// Expectation: success
TEST_F(TestActivationCost, test_CostGeneration) {
  // Currently, the implementation of GetForwardCommCost() method
  // does not check the tensor layouts information, instead, it checks the
  // tensor shape and the slice shape.
  TensorLayout input0_layout, output0_layout;
  Shape input0_shape{200, 300}, output0_shape{200, 300};
  Shape input0_slice_shape{20, 30}, output0_slice_shape{20, 30};
  TensorInfo input0_info(input0_layout, input0_shape, input0_slice_shape),
    output0_info(output0_layout, output0_shape, output0_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input0_info);
  outputs.push_back(output0_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  ac_cost_.set_is_parameter({false, false});
  ac_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  ac_cost_.GetComputationCost(inputs, outputs, 0);
  ac_cost_.GetForwardComputationCost(inputs, outputs, 0);
  ac_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST PReLUCost *********************************************
class TestPReLUCost : public UT::Common {
 public:
  TestPReLUCost() {}
  void SetUp();
  void TearDown();
  PReLUCost prelu_cost_;
};

void TestPReLUCost::SetUp() {
  prelu_cost_ = PReLUCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestPReLUCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for PReLU
/// Description:
/// Expectation: success
TEST_F(TestPReLUCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  prelu_cost_.set_is_parameter({false, true});
  prelu_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  prelu_cost_.GetComputationCost(inputs, outputs, 0);
  double BCC = prelu_cost_.GetBackwardCommCost(inputs, outputs, 0);
  double FMC = prelu_cost_.GetForwardComputationCost(inputs, outputs, 0);
  double GMC = prelu_cost_.GetBackwardComputationCost(inputs, outputs, 0);
  ASSERT_EQ(BCC, 32 * 4);
  ASSERT_EQ(FMC, 8 * 32 * 8 * 8 * 4 + 32 * 4);
  ASSERT_EQ(GMC, 128);
}


//TEST BatchNormCost *********************************
class TestBatchNormCost : public UT::Common {
 public:
  TestBatchNormCost() {}
  void SetUp();
  void TearDown();
  BatchNormCost bn_cost_;
};

void TestBatchNormCost::SetUp() {
  bn_cost_ = BatchNormCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestBatchNormCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for BatchNorm
/// Description:
/// Expectation: success
TEST_F(TestBatchNormCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  bn_cost_.set_is_parameter({false, true});
  bn_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  bn_cost_.GetComputationCost(inputs, outputs, 0);
  bn_cost_.GetBackwardCommCost(inputs, outputs, 0);
  bn_cost_.GetForwardComputationCost(inputs, outputs, 0);
  bn_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST SoftmaxCost **************************************
class TestSoftmaxCost : public UT::Common {
 public:
  TestSoftmaxCost() {}
  void SetUp();
  void TearDown();
  SoftmaxCost sm_cost_;
};

void TestSoftmaxCost::SetUp() {
  sm_cost_ = SoftmaxCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestSoftmaxCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for Softmax
/// Description:
/// Expectation: success
TEST_F(TestSoftmaxCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  sm_cost_.set_is_parameter({false, true});
  sm_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  sm_cost_.GetComputationCost(inputs, outputs, 0);
  sm_cost_.GetBackwardCommCost(inputs, outputs, 0);
  sm_cost_.GetForwardComputationCost(inputs, outputs, 0);
  sm_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST BatchParallelCost  **************************************
class TestBatchParallelCost : public UT::Common {
 public:
  TestBatchParallelCost() {}
  void SetUp();
  void TearDown();
  BatchParallelCost bp_cost_;
};

void TestBatchParallelCost::SetUp() {
  bp_cost_ = BatchParallelCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestBatchParallelCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs in BatchParallel
/// Descrption:
/// Expectation: success
TEST_F(TestBatchParallelCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  bp_cost_.set_is_parameter({false, true});
  bp_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  bp_cost_.GetComputationCost(inputs, outputs, 0);
  bp_cost_.GetBackwardCommCost(inputs, outputs, 0);
  bp_cost_.GetForwardComputationCost(inputs, outputs, 0);
  bp_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST VirtualDatasetCost *************************************
class TestVirtualDatasetCost : public UT::Common {
 public:
  TestVirtualDatasetCost() {}
  void SetUp();
  void TearDown();
  VirtualDatasetCost vd_cost_;
};

void TestVirtualDatasetCost::SetUp() {
  vd_cost_ = VirtualDatasetCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestVirtualDatasetCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for VirtualDataset
/// Description:
/// Expectation: success
TEST_F(TestVirtualDatasetCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  vd_cost_.set_is_parameter({false, true});
  vd_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  vd_cost_.GetComputationCost(inputs, outputs, 0);
  vd_cost_.GetBackwardCommCost(inputs, outputs, 0);
  vd_cost_.GetForwardComputationCost(inputs, outputs, 0);
  vd_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST OneHotCost *************************************
class TestOneHotCost : public UT::Common {
 public:
  TestOneHotCost() {}
  void SetUp();
  void TearDown();
  OneHotCost oh_cost_;
};

void TestOneHotCost::SetUp() {
  oh_cost_ = OneHotCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestOneHotCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for OneHot
/// Description:
/// Expectation: success
TEST_F(TestOneHotCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  oh_cost_.set_is_parameter({false, true});
  oh_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  oh_cost_.GetComputationCost(inputs, outputs, 0);
  oh_cost_.GetBackwardCommCost(inputs, outputs, 0);
  oh_cost_.GetForwardComputationCost(inputs, outputs, 0);
  oh_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}


//TEST SoftmaxCrossEntropyWithLogitsCost *************************************
class TestSoftmaxCrossEntropyWithLogitsCost : public UT::Common {
 public:
  TestSoftmaxCrossEntropyWithLogitsCost() {}
  void SetUp();
  void TearDown();
  SoftmaxCrossEntropyWithLogitsCost scewl_cost_;
};

void TestSoftmaxCrossEntropyWithLogitsCost::SetUp() {
  scewl_cost_ = SoftmaxCrossEntropyWithLogitsCost();
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestSoftmaxCrossEntropyWithLogitsCost::TearDown() {
  // destroy resources
}

/// Feature: test different costs for SoftmaxCrossEntropyWithLogits
/// Description:
/// Expectation: success
TEST_F(TestSoftmaxCrossEntropyWithLogitsCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};

  scewl_cost_.set_is_parameter({false, true});
  scewl_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  scewl_cost_.GetComputationCost(inputs, outputs, 0);
  scewl_cost_.GetBackwardCommCost(inputs, outputs, 0);
  scewl_cost_.GetForwardComputationCost(inputs, outputs, 0);
  scewl_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}
}  // namespace parallel
}  // namespace mindspore
