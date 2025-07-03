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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 4, 2, 2})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({5, 5, 5, 5})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 12, 18, 20})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{4, -1, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 10}}, kNumberTypeFloat32},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 6, 8, 10})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 10, 18, 20})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({3, 3, 6, 8})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
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
      // pre_token_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 4, 5, 3})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{4, 10, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // x
      InferInfoParam{ShapeArray{{10, 20}}, kNumberTypeInt8},
      // weight
      InferInfoParam{ShapeArray{{4, 20, 8}}, kNumberTypeInt8},
      // bias
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // scale
      InferInfoParam{ShapeArray{{4, 8}}, kNumberTypeBFloat16},
      // offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // antiquant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // pre_token_scale
      InferInfoParam{ShapeArray{{10}}, kNumberTypeFloat32},
      // group_list
      InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({2, 4, 2, 2})},
      // activation_input
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_scale
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // activation_quant_offset
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      // split_item
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      // group_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // group_list_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      // act_type
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      // output_dtype
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{10, 8}}, {kNumberTypeBFloat16});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(GroupedMatmulV4, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
