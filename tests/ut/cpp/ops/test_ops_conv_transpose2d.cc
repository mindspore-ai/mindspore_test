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
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 4, 5, 5}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{4, 8, 3, 3}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // stride
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(1)}},
      // paddinng
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(0)}},
      // output_padding
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // dilation
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{1, 8, 7, 7}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{4, 8, 3, 3}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // stride
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(1)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
      // output_padding
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // dilation
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(3)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, 4, 5, 5}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // stride
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(1)}},
      // paddinng
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(0)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{1, 4, 5, 5}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{4, 8, 3, 3}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
      // output_padding
      InferInfoParam{ShapeArray{{}}, TypeIdList{kNumberTypeInt64}, ValuePtrList{CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{1, -1, 7, 7}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(3)}, true},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{24, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{kValueAny, CreateScalar<int64_t>(3)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{24, -1, 47}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), kValueAny}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{24, 79, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(3)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{kValueAny, kValueAny}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{24, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(3)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{kValueAny, kValueAny}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(2)}},
    })
    .FeedExpectedOutput({{24, -1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 20, 15}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 12, 6, 6}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24}, kNumberTypeFloat32},
      // stride
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(3)}},
      // paddinng
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
      // output_padding
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(0)}},
      // groups
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // dilation
      InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{kValueAny, kValueAny}},
    })
    .FeedExpectedOutput({{24, -1, -1}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(ConvTranspose2D, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
