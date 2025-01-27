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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_SOLVE_FLASH_ATTENTION_SCORE_FOR_SEQPIPE_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_SOLVE_FLASH_ATTENTION_SCORE_FOR_SEQPIPE_H_

#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
namespace mindspore {
namespace parallel {
void SolveFASparseForSeqPipe(const CNodePtrList &call_cnode_list, const size_t seq_chunk_num);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_SOLVE_FLASH_ATTENTION_SCORE_FOR_SEQPIPE_H_
