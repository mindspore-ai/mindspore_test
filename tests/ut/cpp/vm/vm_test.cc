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
#include "backend/graph_compiler/vm.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "common/common_test.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace compile {

class TestCompileVM : public UT::Common {
 public:
  TestCompileVM() {}
  virtual ~TestCompileVM() {}

 public:
  virtual void SetUp();
  virtual void TearDown();
};

void TestCompileVM::SetUp() { MS_LOG(INFO) << "TestCompileVM::SetUp()"; }

void TestCompileVM::TearDown() { MS_LOG(INFO) << "TestCompileVM::TearDown()"; }

TEST_F(TestCompileVM, StructPartial) {
  auto partial = new StructPartial(100, VectorRef({20, 60, 100.0}));
  std::stringstream ss;
  ss << *partial;
  delete partial;
  partial = nullptr;
}
}  // namespace compile
}  // namespace mindspore
