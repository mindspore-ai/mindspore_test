/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
      InferInfoParam{ShapeVector{4, 32, 512}, kNumberTypeFloat16},                   // q_nope
      InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeFloat16},                    // q_rope
      InferInfoParam{ShapeVector{1024, 128, 1, 512}, kNumberTypeFloat16},            // ctkv
      InferInfoParam{ShapeVector{1024, 128, 1, 64}, kNumberTypeFloat16},             // k_rope
      InferInfoParam{ShapeVector{4, 128}, kNumberTypeInt32},                         // block_tables
      InferInfoParam{ShapeVector{126, 512}, kNumberTypeInt32},                       // mask
      InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},                            // deq_scale_qk
      InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},                            // deq_scale_pv
      InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                              // q_seq_lens
      InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                              // context_lengths
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},    // q_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.01)},  // scale_value
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},     // kv_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},     // mask_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},     // is_ring
    })
    .FeedExpectedOutput({{4, 32, 512} /* attention_out*/, {0} /* lse_out */}, {kNumberTypeFloat16, kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},                   // q_nope
      InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},                   // q_rope
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},               // ctkv
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},               // k_rope
      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt32},                         // block_tables
      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt32},                         // mask
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},                           // deq_scale_qk
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},                           // deq_scale_pv
      InferInfoParam{ShapeVector{-1}, kNumberTypeInt32},                             // q_seq_lens
      InferInfoParam{ShapeVector{-1}, kNumberTypeInt32},                             // context_lengths
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},    // q_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.01)},  // scale_value
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},     // kv_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},     // mask_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},     // is_ring
    })
    .FeedExpectedOutput({{-1, -1, -1} /* attention_out*/, {0} /* lse_out */}, {kNumberTypeFloat16, kNumberTypeFloat16});

  // MASK_FREE
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{4, 32, 512}, kNumberTypeFloat16},                   // q_nope
      InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeFloat16},                    // q_rope
      InferInfoParam{ShapeVector{1024, 128, 1, 512}, kNumberTypeFloat16},            // ctkv
      InferInfoParam{ShapeVector{1024, 128, 1, 64}, kNumberTypeFloat16},             // k_rope
      InferInfoParam{ShapeVector{4, 128}, kNumberTypeInt32},                         // block_tables
      InferInfoParam{ShapeVector{128, 128}, kNumberTypeInt32},                       // mask
      InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},                            // deq_scale_qk
      InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},                            // deq_scale_pv
      InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                              // q_seq_lens
      InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                              // context_lengths
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},    // q_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.01)},  // scale_value
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},     // kv_head_num
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},     // mask_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},     // is_ring
    })
    .FeedExpectedOutput({{4, 32, 512} /* attention_out*/, {0} /* lse_out */}, {kNumberTypeFloat16, kNumberTypeFloat16});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Mla, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
