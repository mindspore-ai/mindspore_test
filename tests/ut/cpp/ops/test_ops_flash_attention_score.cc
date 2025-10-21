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

#include <vector>
#include <memory>

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "mindspore/ops/op_def/op_name.h"
#include "infer/ops_func_impl/flash_attention_score.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "ops/utils/general_infer_param.h"
#include "ops/op_def.h"
#include "ir/value.h"

namespace mindspore {
namespace ops {

// Forward declarations
static std::vector<GeneralInferParam> prepare_params_flash_attention();
static std::vector<GeneralInferParam> prepare_params_flash_attention_exceptions();

constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

static std::vector<GeneralInferParam> prepare_params_flash_attention() {
  GeneralInferParamGenerator gen;

  // Case 0: BSH, fp16, keep_prob=1.0, outputs: [B,N1,S1,8], [B,N1,S1,8], [1], [B,S1,H1]
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;  // H1=N1*D=32
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // real_shift=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // drop_mask=None (keep_prob=1)
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // padding_mask=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // attn_mask=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // prefix=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_qlen=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_kvlen=None
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // sparse_mode=0
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 1: BSH, bfloat16, keep_prob=1.0
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeBFloat16, kNumberTypeBFloat16});
  }

  // Case 2: BSH with attn_mask bool [S1, S1]
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{S1, S1}, kNumberTypeBool},  // attn_mask bool
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 3: BSH, known rank with dynamic dims [-1,-1,-1]
  {
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{-1, 1, -1, 8}, ShapeVector{-1, 1, -1, 8}, ShapeVector{1},
                            ShapeVector{-1, -1, -1}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 4: TND, fp16, actual_seq_qlen/kvlen as tuple (ValuePtrList cumulative)
  {
    const int64_t T = 128, N = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                        ValuePtrList{CreateScalar<int64_t>(32), CreateScalar<int64_t>(64),
                                     CreateScalar<int64_t>(96), CreateScalar<int64_t>(128)}},
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                        ValuePtrList{CreateScalar<int64_t>(32), CreateScalar<int64_t>(64),
                                     CreateScalar<int64_t>(96), CreateScalar<int64_t>(128)}},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // TND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, N, 8}, ShapeVector{T, N, 8}, ShapeVector{1}, ShapeVector{T, N, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 5: BNSD, fp16
  {
    const int64_t B = 2, N1 = 4, S1 = 16, D = 8, N2 = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, N1, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, N2, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, N2, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},  // BNSD
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, N1, S1, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 6: SBH, fp16
  {
    const int64_t S1 = 16, B = 2, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // SBH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{S1, B, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 7: BSND, fp16
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8, N2 = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N2, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N2, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},  // BSND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 8: TND, fp16, softmax [T,N,8]; attention [T,N,D]
  {
    const int64_t T = 128, N = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // actual_seq_qlen / kvlen will be provided as scalars below
         // actual_seq_qlen / kvlen (scalars or arrays of cumulative)
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // TND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, N, 8}, ShapeVector{T, N, 8}, ShapeVector{1}, ShapeVector{T, N, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 9: TH, fp16, softmax [T, head_num*8]; attention [T, H]
  {
    const int64_t T = 128, H = 64, head_num = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(head_num)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},  // TH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, head_num * 8}, ShapeVector{T, head_num * 8}, ShapeVector{1},
                            ShapeVector{T, H}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 10: Dynamic rank query, BSH
  {
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{kShapeDimAny, kShapeDimAny, kShapeDimAny, 8}, ShapeVector{kShapeDimAny, kShapeDimAny, kShapeDimAny, 8},
                            ShapeVector{1}, ShapeVector{kShapeRankAny}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  return gen.Generate();
}
static std::vector<GeneralInferParam> prepare_params_flash_attention_exceptions() {
  GeneralInferParamGenerator gen;

  // case 11: keep_prob=1.0 with drop_mask provided -> expect throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{1, 1, 1, 1}, kNumberTypeUInt8},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 12: keep_prob in (0,1) but drop_mask dtype != uint8 -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{1, 1, 1, 1}, kNumberTypeFloat16},  // wrong dtype
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.5f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 13: keep_prob out of range -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.5f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 14: padding_mask not None -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{1, 1, 1, 1}, kNumberTypeUInt8},  // padding_mask not none
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 15: TND missing actual_seq_qlen/kvlen -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{128, 2, 64}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{128, 2, 64}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{128, 2, 64}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_qlen None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_kvlen None
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 16: BSH hidden_size % head_num != 0 -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{2, 16, 30}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 16, 30}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 16, 30}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeArray{{}}, kNumberTypeFloat16, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 17: BNSD query.shape[1] != head_num -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{2, 4, 16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 2, 16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 2, 16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 18: SBH rank of q/key must be 3 -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{16, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 19: drop_mask shape wrong S2/8 -> throw
  gen.FeedInputArgs({
       InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{2, 4, 16, 3}, kNumberTypeUInt8},  // wrong last dim
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.5f)},
       InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
     })
     .CaseShouldThrow();

  // case 20: prefix mode=5 ok; non-5 with prefix -> throw
  {
    // ok
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{2048, 2048}, kNumberTypeBool},
         // prefix sequence length B=2
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
          ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(1)}},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
       })
       .FeedExpectedOutput({ShapeVector{2, 4, 16, 8}, ShapeVector{2, 4, 16, 8}, ShapeVector{1},
                            ShapeVector{2, 16, 32}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }
  {
    // case 21: not mode=5 with prefix -> throw
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
          ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(1)}},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .CaseShouldThrow();
  }

  return gen.Generate();
}

// Merge all params into a single vector to ensure op_type resolves to 'FlashAttentionScore'
static std::vector<GeneralInferParam> prepare_params_flash_attention_all() {
  auto success_cases = prepare_params_flash_attention();
  auto exception_cases = prepare_params_flash_attention_exceptions();
  success_cases.insert(success_cases.end(), exception_cases.begin(), exception_cases.end());
  return success_cases;
}

INSTANTIATE_TEST_CASE_P(FlashAttentionScore, GeneralInferTest,
                        testing::ValuesIn(prepare_params_flash_attention_all()));

}  // namespace ops
}  // namespace mindspore
