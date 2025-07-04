/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_sequential_sample.h"

namespace mindspore {
namespace mindrecord {
ShardSequentialSample::ShardSequentialSample(int64_t n, int64_t offset)
    : ShardSample(n), offset_(offset), per_(0.0f), per_offset_(0.0f) {}

ShardSequentialSample::ShardSequentialSample(float per, float per_offset)
    : ShardSample(0), offset_(0), per_(per), per_offset_(per_offset) {}

int64_t ShardSequentialSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (offset_ > 0) {
    dataset_size -= offset_;
  }
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    return dataset_size;
  }
  if (per_ > kEpsilon && per_ <= 1.0f) {
    return dataset_size * kEpsilon;
  }
  return std::min(static_cast<int64_t>(no_of_samples_), dataset_size);
}

Status ShardSequentialSample::Execute(ShardTaskList &tasks) {
  int64_t taking;
  int64_t total_no = tasks.SizeAfterSampling();
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    taking = total_no;
  } else if (per_ > kEpsilon && per_ <= 1.0f) {
    taking = total_no * kEpsilon;
  } else {
    taking = std::min(static_cast<int64_t>(no_of_samples_), total_no);
  }
  if (tasks.load_mode_ != LoadMode::kSlow) {
    if (tasks.permutation_.empty()) {
      ShardTaskList new_tasks;
      total_no = static_cast<int64_t>(tasks.Size());
      if (no_of_samples_ != 0) {
        taking = taking + offset_ > total_no ? total_no : taking + offset_;
      }
      for (int64_t i = offset_; i < taking; ++i) {
        CHECK_FAIL_RETURN_UNEXPECTED_MR(total_no != 0, "Divisor 'total_no' is zero.");
        new_tasks.AssignTask(tasks, i % total_no);
      }
      ShardTaskList::TaskListSwap(tasks, new_tasks);
    } else {  // shuffled
      ShardTaskList new_tasks;
      total_no = static_cast<int64_t>(tasks.permutation_.size());
      if (no_of_samples_ != 0) {
        taking = taking + offset_ > total_no ? total_no : taking + offset_;
      }
      for (int64_t i = offset_; i < taking; ++i) {
        CHECK_FAIL_RETURN_UNEXPECTED_MR(total_no != 0, "Divisor 'total_no' is zero.");
        new_tasks.AssignTask(tasks, tasks.permutation_[i % total_no]);
      }
      ShardTaskList::TaskListSwap(tasks, new_tasks);
    }
  } else {
    // update the partitioned_shard_sample_count_ by no_of_samples
    tasks.UpdatePartitionedShardSampleCountByNumSamples(taking);
  }
  return Status::OK();
}

}  // namespace mindrecord
}  // namespace mindspore
