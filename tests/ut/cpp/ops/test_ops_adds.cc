/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32}})
    .FeedExpectedOutput({{3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32}})
    .FeedExpectedOutput({{3, 2}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32}})
    .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeDouble},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeDouble});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AddScalar, GeneralInferTest, testing::ValuesIn(prepare_params()));

struct AddExtInferValueParams {
  tensor::TensorPtr input;
  tensor::TensorPtr other;
  ValuePtr alpha;
  tensor::TensorPtr out;
};

class TestAddExtInferValue : public TestOps, public testing::WithParamInterface<AddExtInferValueParams> {};

TEST_P(TestAddExtInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.other, nullptr);
  auto other = param.other->ToAbstract();
  ASSERT_NE(other, nullptr);

  ASSERT_NE(param.alpha, nullptr);
  auto alpha = param.alpha->ToAbstract();
  ASSERT_NE(alpha, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, other, alpha};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimAddExt, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "AddExt have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "AddExt can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestAddExtInferValue, TestAddExtInferValue,
  testing::Values(
    AddExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1,2}, std::vector<float>{2.0, 3.0}),
                           CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1,2}, std::vector<float>{4.0, 6.0}),
                           CreateScalar<int64_t>(1),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2}, std::vector<float>{6.0, 9.0})}));


struct AddScalarInferValueParams {
    tensor::TensorPtr input;
    ValuePtr other;
    ValuePtr alpha;
    tensor::TensorPtr out;
};

class TestAddScalarInferValue : public TestOps, public testing::WithParamInterface<AddScalarInferValueParams> {};

TEST_P(TestAddScalarInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.other, nullptr);
  auto other = param.other->ToAbstract();
  ASSERT_NE(other, nullptr);

  ASSERT_NE(param.alpha, nullptr);
  auto alpha = param.alpha->ToAbstract();
  ASSERT_NE(alpha, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, other, alpha};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimAddScalar, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "AddScalar have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "AddScalar can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
        TestAddScalarInferValue, TestAddScalarInferValue,
        testing::Values(
        AddScalarInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1,2}, std::vector<float>{2.0, 3.0}),
                               CreateScalar<int64_t>(2),
                               CreateScalar<int64_t>(1),
                               CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2}, std::vector<float>{4.0, 5.0})}));
}  // namespace mindspore::ops
