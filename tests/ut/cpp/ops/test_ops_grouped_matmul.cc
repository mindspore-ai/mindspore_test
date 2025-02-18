/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
      // x
      InferInfoParam{ShapeArray{{2, 4, 8}, {1, 5}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{8, 10}, {5, 3}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{2, 4, 10}, {1, 3}}, {kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-2}, {1, 5}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{8, 10}, {5, 3}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-2}, {1, 3}}, {kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-1, -1, 8}, {1, 5}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{8, 10}, {5, 3}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-1, -1, 10}, {1, 3}}, {kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{2, 4, -1}, {1, -1}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{-1, 10}, {-1, 3}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{2, 4, 10}, {1, 3}}, {kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{4, 20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-2}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{4, 20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-1, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, -1}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{4, -1, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-1, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{4, 20, -1}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 12, 20})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, 10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-2}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 10, 18, 20})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, -1, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{-2}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 15, 20})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, 10, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-1, 10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{-1, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 10, 13, 20})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, -1, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, -1}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 13, 17, 20})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, 10, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{20, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{-1, 10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, -1}}, kNumberTypeFloat32},
      // weight
      InferInfoParam{ShapeArray{{-1, 8}}, kNumberTypeFloat32},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // transpose_a
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
      // transpose_b
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(False)},
    })
    .FeedExpectedOutput({{4, 10, 8}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(GroupedMatmul, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
