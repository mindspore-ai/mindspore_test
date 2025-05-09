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

#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  // input.dtype=int64, size=(3, 4), dtype=float32
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt64},
      InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                     ValuePtrList{CreateScalar<int64_t>(3), CreateScalar<int64_t>(4)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat32)},
    })
    .FeedExpectedOutput({{3, 4}}, {kNumberTypeFloat32});
  // input.dtype=int64, size=(3), dtype=bool
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt64},
      InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(3)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<bool>(false)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeBool)},
    })
    .FeedExpectedOutput({{3}}, {kNumberTypeBool});
  // input.dtype=int64, size=(0), dtype=bfloat16
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt64},
      InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(0)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.0)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeBFloat16)},
    })
    .FeedExpectedOutput({{0}}, {kNumberTypeBFloat16});
  // input.dtype=int32, size=(3, 0), dtype=int64
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt32},
      InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                     ValuePtrList{CreateScalar<int64_t>(3), CreateScalar<int64_t>(0)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<int64_t>(2)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt64)},
    })
    .FeedExpectedOutput({{3, 0}}, {kNumberTypeInt64});
  // input.dtype=int64, size=(), dtype=uint8
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt64},
      InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<uint8_t>(5)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeUInt8)},
    })
    .FeedExpectedOutput({{}}, {kNumberTypeUInt8});
  // input.dtype=int64, size=(-1), dtype=float32
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt64},
      InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(-1)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.1)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat32)},
    })
    .CaseShouldThrow();
  // input.dtype=int16, size=(3, 4), dtype=None
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 2}, kNumberTypeInt16},
      InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                     ValuePtrList{CreateScalar<int64_t>(3), CreateScalar<int64_t>(4)}},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<int16_t>(10)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64},
    })
    .FeedExpectedOutput({{3, 4}}, {kNumberTypeInt16});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(NewFull, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
