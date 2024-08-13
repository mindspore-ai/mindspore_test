/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/zeros.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "ops/test_ops_cmp_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct ZerosShapeParam {
  ValuePtr shape_value;
  ValuePtr dtype_value;
  ShapeVector output_shape;
  TypePtr output_dtype;
};

// Test InferShape and InferType
class TestZeros : public TestOps, public testing::WithParamInterface<ZerosShapeParam> {};

TEST_P(TestZeros, zeros_dyn_shape) {
  const auto param = GetParam();
  auto shape_list = param.shape_value->cast<ValueTuplePtr>();
  ASSERT_NE(shape_list, nullptr);
  auto shape_dim = shape_list->size();
  auto in_shape = param.shape_value->ToAbstract();
  auto in_shape_seq = in_shape->cast<abstract::AbstractSequencePtr>();
  if (shape_dim == 0) {
    in_shape_seq->CheckAndConvertToDynamicLenSequence();
  }
  auto dtype_value = param.dtype_value->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<ZerosFuncImpl>(kNameZeros, {in_shape, dtype_value}, expect_shape, expect_type);
}

auto zeros_test_cases = testing::Values(
  ZerosShapeParam{CreatePyIntTuple({2, 2, 3}), CreateScalar<int64_t>(kNumberTypeInt64), {2, 2, 3}, kInt64},
  ZerosShapeParam{CreatePyIntTuple({3, 4}), CreateScalar<int64_t>(kNumberTypeFloat32), {3, 4}, kFloat32},
  ZerosShapeParam{CreatePyIntTuple({kValueAny, 2, 3}), CreateScalar<int64_t>(kNumberTypeFloat64), {-1, 2, 3}, kFloat64},
  ZerosShapeParam{CreatePyIntTuple({}), CreateScalar<int64_t>(kNumberTypeInt32), {-2}, kInt32});
INSTANTIATE_TEST_CASE_P(TestZeros, TestZeros, zeros_test_cases);

// Test InferValue
class TestZerosInferValue : public TestOps, public testing::WithParamInterface<ZerosShapeParam> {};

TEST_P(TestZerosInferValue, zeros_infer_value) {
  const auto param = GetParam();
  auto shape_list = param.shape_value->cast<ValueTuplePtr>();
  ASSERT_NE(shape_list, nullptr);
  auto shape_dim = shape_list->size();
  auto in_shape = param.shape_value->ToAbstract();
  auto in_shape_seq = in_shape->cast<abstract::AbstractSequencePtr>();
  if (shape_dim == 0) {
    in_shape_seq->CheckAndConvertToDynamicLenSequence();
  }
  auto dtype_value = param.dtype_value->ToAbstract();
  auto input_args = abstract::AbstractBasePtrList{in_shape, dtype_value};
  auto out_value_opt = abstract::InferValueByFuncImpl(prim::kPrimZeros, input_args);
  if (!out_value_opt.has_value()) {
    MS_LOG(ERROR) << "Zeros has no inferValue implement!";
    ASSERT_TRUE(False);
  }
  auto infer_out = out_value_opt.value();
  ASSERT_NE(infer_out, nullptr);
  auto tensor_out = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(tensor_out, nullptr);
  auto expect_out = TensorConstructUtils::CreateZerosTensor(param.output_dtype, param.output_shape);
  ASSERT_TRUE(tensor_out->ValueEqual(*expect_out));
}

auto zeros_infer_value_test_cases = testing::Values(
  ZerosShapeParam{CreatePyIntTuple({2, 2, 3}), CreateScalar<int64_t>(kNumberTypeInt64), {2, 2, 3}, kInt64},
  ZerosShapeParam{CreatePyIntTuple({3, 4}), CreateScalar<int64_t>(kNumberTypeFloat32), {3, 4}, kFloat32});
INSTANTIATE_TEST_CASE_P(TestZerosInferValue, TestZerosInferValue, zeros_infer_value_test_cases);
}  // namespace ops
}  // namespace mindspore
