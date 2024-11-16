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

#include "common/common_test.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/index.h"


namespace mindspore {
namespace ops {
struct IndexOpParamShape {
  ShapeVector input_shape;
  std::vector<ShapeVector> indices_shapes;
  ShapeVector output_shape;
};

struct IndexOpParamType {
  TypePtr input_type;
  TypePtr indices_type;
  TypePtr output_type;
};
class TestIndexSimpleInfer : public TestOps,
                             public testing::WithParamInterface<std::tuple<IndexOpParamShape, IndexOpParamType>> {};

TEST_P(TestIndexSimpleInfer, simple_infer) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto input = std::make_shared<tensor::BaseTensor>(dtype_param.input_type->type_id(), shape_param.input_shape);
  std::vector<ValuePtr> indices;
  for (auto indice_shape : shape_param.indices_shapes) {
    auto tmp_tensor = std::make_shared<tensor::BaseTensor>(dtype_param.indices_type->type_id(), indice_shape);
    indices.push_back(tmp_tensor);
  }
  ValuePtrList input_values;
  input_values.push_back(std::move(input));
  input_values.push_back(std::make_shared<ValueTuple>(indices));

  IndexFuncImpl index_func_impl;
  auto prim = std::make_shared<Primitive>("Index");
  auto expect_shape = ShapeArray{shape_param.output_shape};
  auto expect_type = TypePtrList{dtype_param.output_type};

  auto output_shape = index_func_impl.InferShape(prim, input_values);
  auto output_type = index_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto IndexOpSimpleInferShapeTestCases = testing::ValuesIn({
  IndexOpParamShape{{4, 2, 3}, {{2, 2}, {1, 2}}, {2, 2, 3}},
  IndexOpParamShape{{2, 4, 3}, {{2, 2, 2}, {2, 2}}, {2, 2, 2, 3}},
  IndexOpParamShape{{10, 4, 2}, {{2, 3}, {3, 1, 1}, {3}}, {3, 2, 3}},
});

auto IndexOpTypeTestCases =
  testing::ValuesIn({IndexOpParamType{kBool, kInt64, kBool}, IndexOpParamType{kFloat16, kInt64, kFloat16},
                     IndexOpParamType{kFloat32, kInt64, kFloat32}, IndexOpParamType{kFloat64, kInt64, kFloat64},
                     IndexOpParamType{kUInt8, kInt64, kUInt8}, IndexOpParamType{kInt8, kInt64, kInt8},
                     IndexOpParamType{kInt16, kInt64, kInt16}, IndexOpParamType{kInt32, kInt64, kInt32},
                     IndexOpParamType{kInt64, kInt64, kInt64}, IndexOpParamType{kBFloat16, kInt64, kBFloat16}});

INSTANTIATE_TEST_CASE_P(TestIndexSimpleInfer, TestIndexSimpleInfer,
                        testing::Combine(IndexOpSimpleInferShapeTestCases, IndexOpTypeTestCases));

}  // namespace ops
}  // namespace mindspore
