/*
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_sample.h"

#include "utils/ms_utils.h"

namespace mindspore {
namespace mindrecord {
ShardSample::ShardSample(int64_t n)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(n),
      indices_({}),
      sampler_type_(kCustomTopNSampler),
      offset_(-1) {}

ShardSample::ShardSample(int64_t num, int64_t den)
    : numerator_(num),
      denominator_(den),
      partition_id_(0),
      no_of_samples_(0),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(-1) {}

ShardSample::ShardSample(int64_t num, int64_t den, int64_t par, int64_t no_of_samples, int64_t offset)
    : numerator_(num),
      denominator_(den),
      partition_id_(par),
      no_of_samples_(no_of_samples),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(offset) {}

ShardSample::ShardSample(const std::vector<int64_t> &indices)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(0),
      indices_(indices),
      sampler_type_(kSubsetSampler) {}

ShardSample::ShardSample(const std::vector<int64_t> &indices, uint32_t seed) : ShardSample(indices) {
  sampler_type_ = kSubsetRandomSampler;
  shuffle_op_ = std::make_shared<ShardShuffle>(seed);
}

int64_t ShardSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (sampler_type_ == kCustomTopNSampler) {
    return no_of_samples_;
  }

  if (sampler_type_ == kCustomTopPercentSampler) {
    if (dataset_size % denominator_ == 0) {
      return dataset_size / denominator_ * numerator_;
    } else {
      return dataset_size / denominator_ * numerator_ + 1;
    }
  }
  if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
    return indices_.size();
  }
  return 0;
}

namespace {
inline void ComputeDistributed(int64_t partition_id, int64_t no_of_samples, int64_t denominator, int64_t taking,
                               ShardTaskList *tasks, ShardTaskList *new_tasks) {
  int64_t count = 0;
  auto total_no = tasks->sample_ids_.size();
  std::string env_mindrecord_shard_by_block = common::GetEnv("MS_DEV_MINDRECORD_SHARD_BY_BLOCK");
  (void)transform(env_mindrecord_shard_by_block.begin(), env_mindrecord_shard_by_block.end(),
                  env_mindrecord_shard_by_block.begin(), ::tolower);
  if (env_mindrecord_shard_by_block == "true") {
    // Distributed by block
    for (int64_t i = partition_id * taking; i < (partition_id + 1) * taking; i++) {
      if (no_of_samples != 0 && count == no_of_samples) {
        break;
      }
      new_tasks->AssignTask(*tasks, i % total_no);  // rounding up. if overflow, go back to start
      count++;
    }
  } else {
    // Distributed by slice
    for (int64_t i = 0; i < taking; i++) {
      if (no_of_samples != 0 && count == no_of_samples) {
        break;
      }
      // Sampling logic formula by slice: sample_ids = partition_id + denominator * index
      // partition_id represents the card number of the current shard
      // denominator represents the number of shards
      // index represents the index of the sample in the card number
      new_tasks->AssignTask(*tasks,
                            (partition_id + denominator * i) % total_no);  // rounding up. if overflow, go back to start
      count++;
    }
  }
}

inline void AddToPssc(PartitionedShardSampleCount *partitioned_shard_sample_count,
                      std::vector<int64_t> tasks_shard_sample_count, int32_t shard_index, int64_t current_sample_id,
                      int64_t num_samples_per_partition, int64_t sample_count, int64_t denominator) {
  partitioned_shard_sample_count->task_type = TaskType::kCommonTask;
  partitioned_shard_sample_count->shard_id = shard_index;
  partitioned_shard_sample_count->start = current_sample_id;
  partitioned_shard_sample_count->end =
    (current_sample_id + (num_samples_per_partition - sample_count) * denominator) >
        tasks_shard_sample_count[shard_index]
      ? tasks_shard_sample_count[shard_index]
      : (current_sample_id + (num_samples_per_partition - sample_count) * denominator);
  partitioned_shard_sample_count->step = denominator;
}
}  // namespace

Status ShardSample::UpdateTasks(ShardTaskList &tasks, int64_t taking) {
  if (tasks.permutation_.empty()) {
    ShardTaskList new_tasks;
    auto total_no = tasks.sample_ids_.size();
    CHECK_FAIL_RETURN_UNEXPECTED_MR(
      total_no > 0, "[Internal ERROR] 'total_no' should be positive but got: " + std::to_string(total_no));
    if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
      for (int64_t i = 0; i < indices_.size(); ++i) {
        int64_t index = ((indices_[i] % total_no) + total_no) % total_no;
        new_tasks.AssignTask(tasks, index);  // different mod result between c and python
      }
    } else {
      if (nums_per_shard_.empty()) {
        ComputeDistributed(partition_id_, no_of_samples_, denominator_, taking, &tasks, &new_tasks);
      } else {
        int64_t count = 0;
        // Get samples within a specific range
        int64_t i = partition_id_ - 1 >= 0 ? nums_per_shard_[partition_id_ - 1] : 0;
        for (; i < nums_per_shard_[partition_id_]; i++) {
          if (no_of_samples_ != 0 && count == no_of_samples_) {
            break;
          }
          new_tasks.AssignTask(tasks, i % total_no);
          count++;
        }
      }
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  } else {
    ShardTaskList new_tasks;
    int64_t total_no = tasks.permutation_.size();
    CHECK_FAIL_RETURN_UNEXPECTED_MR(
      total_no > 0, "[Internal ERROR] 'total_no' should be positive but got: " + std::to_string(total_no));
    int64_t cnt = 0;
    for (int64_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
      if (no_of_samples_ != 0 && cnt == no_of_samples_) {
        break;
      }
      new_tasks.AssignTask(tasks, tasks.permutation_[i % total_no]);
      cnt++;
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  }
  return Status::OK();
}

Status ShardSample::UpdatePartitionWhenSlowMode(ShardTaskList &tasks) {
  // distribtued sample when load mode is slow load
  // split shard sample
  // 0 : 17  -  shard0 has 17 samples - pre shard 2
  // 1 : 32  -  shard1 has 15 samples - pre shard 0
  // 2 : 58  -  shard2 has 26 samples - pre shard 1
  // padded_sample = 6
  // Assuming this is an 8-card training
  // card 0 : kCommonTask, 0, 0, 8
  // card 1 : kCommonTask, 0, 8, 16
  // card 2 : kCommonTask, 0, 16, 17
  // card 2 : kCommonTask, 1, 17, 24
  // card 3 : kCommonTask, 1, 24, 32
  // card 4 : kCommonTask, 2, 32, 40
  // card 5 : kCommonTask, 2, 40, 48
  // card 6 : kCommonTask, 2, 48, 56
  // card 7 : kCommonTask, 2, 56, 58
  // card 7 : kPaddedTask, -1, 58, 64
  auto tasks_shard_sample_count = tasks.shuffled_shard_sample_count_;
  int64_t total_sample = tasks_shard_sample_count[tasks_shard_sample_count.size() - 1] + tasks.padded_sample_;
  int64_t step = total_sample % denominator_ == 0 ? total_sample / denominator_ : total_sample / denominator_ + 1;
  int64_t start = partition_id_ * step;
  int64_t end = (partition_id_ + 1) * step;
  std::vector<PartitionedShardSampleCount> vpssc;
  int64_t tmp_start = start;
  int64_t tmp_end = end;
  for (int32_t shard_index = 0; shard_index < tasks_shard_sample_count.size(); shard_index++) {
    if (tmp_start >= tasks_shard_sample_count[shard_index]) {
      continue;
    }

    if (tmp_end <= tasks_shard_sample_count[shard_index]) {
      tmp_end = end;
      // add new range to vp
      PartitionedShardSampleCount pssc;
      pssc.task_type = TaskType::kCommonTask;
      pssc.shard_id = shard_index;
      pssc.start = tmp_start;
      pssc.end = tmp_end;
      vpssc.push_back(pssc);
      break;
    } else {
      PartitionedShardSampleCount pssc;
      pssc.task_type = TaskType::kCommonTask;
      pssc.shard_id = shard_index;
      pssc.start = tmp_start;
      pssc.end = tasks_shard_sample_count[shard_index];
      vpssc.push_back(pssc);
      tmp_start = tasks_shard_sample_count[shard_index];
    }
  }

  // retrieve from the start or padded sample
  if (end > tasks_shard_sample_count[tasks_shard_sample_count.size() - 1]) {
    // padded scenario
    if (tasks.padded_sample_ > 0) {
      if (end - tasks_shard_sample_count[tasks_shard_sample_count.size() - 1] <= tasks.padded_sample_) {
        PartitionedShardSampleCount pssc;
        pssc.task_type = TaskType::kPaddedTask;
        pssc.shard_id = -1;
        pssc.start = tmp_start;
        pssc.end = end;
        vpssc.push_back(pssc);
      } else {
        RETURN_STATUS_UNEXPECTED_MR(
          "It's padded sample scenario, but the total sample: " + std::to_string(total_sample) +
          " which is not divisible by " + std::to_string(denominator_));
      }
    } else {
      tmp_start = 0;
      end = end - tasks_shard_sample_count[tasks_shard_sample_count.size() - 1];
      tmp_end = end;
      for (int32_t shard_index = 0; shard_index < tasks_shard_sample_count.size(); shard_index++) {
        if (tmp_start >= tasks_shard_sample_count[shard_index]) {
          continue;
        }

        if (tmp_end <= tasks_shard_sample_count[shard_index]) {
          tmp_end = end;
          // add new range to vp
          PartitionedShardSampleCount pssc;
          pssc.task_type = TaskType::kCommonTask;
          pssc.shard_id = shard_index;
          pssc.start = tmp_start;
          pssc.end = tmp_end;
          vpssc.push_back(pssc);
          break;
        } else {
          PartitionedShardSampleCount pssc;
          pssc.task_type = TaskType::kCommonTask;
          pssc.shard_id = shard_index;
          pssc.start = tmp_start;
          pssc.end = tasks_shard_sample_count[shard_index];
          vpssc.push_back(pssc);
          tmp_start = tasks_shard_sample_count[shard_index];
        }
      }
    }
  }

  for (const auto &item : vpssc) {
    MS_LOG(INFO) << "Distribute samples, partition_id: " << partition_id_ << ", task_type: " << item.task_type
                 << ", shard_id: " << item.shard_id << ", start: " << item.start << ", end: " << item.end;
  }

  tasks.SetPartitionedShardSampleCount(vpssc);

  // update vpssc by no_of_samples_
  if (no_of_samples_ != 0) {
    tasks.UpdatePartitionedShardSampleCountByNumSamples(no_of_samples_);
  }

  return Status::OK();
}

Status ShardSample::UpdatePartitionWhenSlowModeBySlice(ShardTaskList &tasks) {
  // distribtued sample by slice when load mode is slow load
  // mindrecord files
  // mindrecord file 0 has 50 samples
  // mindrecord file 1 has 40 samples
  // mindrecord file 2 has 60 samples
  // mindrecord file 3 has 30 samples
  // Assuming this is an 3-card training
  // card 0 : kCommonTask, shard_id=0, start=0, end=50, step=3
  // card 0 : kCommonTask, shard_id=1, start=51, end=90, step=3
  // card 0 : kCommonTask, shard_id=2, start=90, end=150, step=3
  // card 0 : kCommonTask, shard_id=3, start=150, end=180, step=3
  // card 1 : kCommonTask, shard_id=0, start=1, end=50, step=3
  // card 1 : kCommonTask, shard_id=1, start=52, end=90, step=3
  // card 1 : kCommonTask, shard_id=2, start=91, end=150, step=3
  // card 1 : kCommonTask, shard_id=3, start=151, end=180, step=3
  // ...
  auto tasks_shard_sample_count = tasks.shuffled_shard_sample_count_;
  int64_t total_sample = tasks_shard_sample_count[tasks_shard_sample_count.size() - 1] + tasks.padded_sample_;
  int64_t num_samples_per_partition =
    total_sample % denominator_ == 0 ? total_sample / denominator_ : total_sample / denominator_ + 1;
  std::vector<PartitionedShardSampleCount> vpssc;
  int64_t current_sample_id = partition_id_;
  // record how many sample ids have been saved
  int64_t sample_count = 0;
  for (int32_t shard_index = 0; shard_index < tasks_shard_sample_count.size(); shard_index++) {
    if (current_sample_id >= tasks_shard_sample_count[shard_index]) {
      continue;
    }
    // how many sample ids are there in the temporary record
    int64_t shard_sample_count = (tasks_shard_sample_count[shard_index] - current_sample_id) % denominator_ == 0
                                   ? (tasks_shard_sample_count[shard_index] - current_sample_id) / denominator_
                                   : (tasks_shard_sample_count[shard_index] - current_sample_id) / denominator_ + 1;
    if ((num_samples_per_partition - sample_count) <= shard_sample_count) {
      // add new range to vp
      PartitionedShardSampleCount pssc;
      AddToPssc(&pssc, tasks_shard_sample_count, shard_index, current_sample_id, num_samples_per_partition,
                sample_count, denominator_);
      vpssc.push_back(pssc);
      // sample_count and current_sample_id need consider uneven distribution scenarios
      current_sample_id += ((num_samples_per_partition - sample_count) * denominator_);
      sample_count += shard_sample_count;
      break;
    } else {
      PartitionedShardSampleCount pssc;
      pssc.task_type = TaskType::kCommonTask;
      pssc.shard_id = shard_index;
      pssc.start = current_sample_id;
      pssc.end = tasks_shard_sample_count[shard_index];
      pssc.step = denominator_;
      vpssc.push_back(pssc);
      current_sample_id += (shard_sample_count * denominator_);
      sample_count += shard_sample_count;
    }
  }
  // retrieve from the start or padded sample
  if ((num_samples_per_partition * denominator_) > tasks_shard_sample_count[tasks_shard_sample_count.size() - 1]) {
    if (sample_count < num_samples_per_partition) {
      // padded scenario
      if (tasks.padded_sample_ > 0) {
        if (total_sample - tasks_shard_sample_count[tasks_shard_sample_count.size() - 1] <= tasks.padded_sample_) {
          PartitionedShardSampleCount pssc;
          pssc.task_type = TaskType::kPaddedTask;
          pssc.shard_id = -1;
          pssc.start = current_sample_id;
          pssc.end = total_sample;
          pssc.step = denominator_;
          vpssc.push_back(pssc);
        } else {
          RETURN_STATUS_UNEXPECTED_MR(
            "It's padded sample scenario, but the total sample: " + std::to_string(total_sample) +
            " which is not divisible by " + std::to_string(denominator_));
        }
      } else {
        // need to recalculate the value of current_sample_id
        if (current_sample_id >= total_sample) {
          current_sample_id -= total_sample;
        }
        for (int32_t shard_index = 0; shard_index < tasks_shard_sample_count.size(); shard_index++) {
          if (current_sample_id >= tasks_shard_sample_count[shard_index]) {
            continue;
          }
          int64_t shard_sample_count =
            (tasks_shard_sample_count[shard_index] - current_sample_id) % denominator_ == 0
              ? (tasks_shard_sample_count[shard_index] - current_sample_id) / denominator_
              : (tasks_shard_sample_count[shard_index] - current_sample_id) / denominator_ + 1;
          if ((num_samples_per_partition - sample_count) <= shard_sample_count) {
            // add new range to vp
            PartitionedShardSampleCount pssc;
            AddToPssc(&pssc, tasks_shard_sample_count, shard_index, current_sample_id, num_samples_per_partition,
                      sample_count, denominator_);
            vpssc.push_back(pssc);
            break;
          } else {
            PartitionedShardSampleCount pssc;
            pssc.task_type = TaskType::kCommonTask;
            pssc.shard_id = shard_index;
            pssc.start = current_sample_id;
            pssc.end = tasks_shard_sample_count[shard_index];
            pssc.step = denominator_;
            vpssc.push_back(pssc);
            current_sample_id += (shard_sample_count * denominator_);
            sample_count += shard_sample_count;
          }
        }
      }
    }
  }
  tasks.SetPartitionedShardSampleCount(vpssc);
  // update vpssc by no_of_samples_
  if (no_of_samples_ != 0) {
    tasks.UpdatePartitionedShardSampleCountByNumSamples(no_of_samples_);
  }
  return Status::OK();
}

Status ShardSample::Execute(ShardTaskList &tasks) {
  if (tasks.load_mode_ != LoadMode::kSlow) {
    if (offset_ != -1) {
      int64_t old_v = 0;
      int64_t num_rows_ = tasks.sample_ids_.size();
      for (int64_t x = 0; x < denominator_; x++) {
        int64_t samples_per_buffer_ = (num_rows_ + offset_) / denominator_;
        int64_t remainder = (num_rows_ + offset_) % denominator_;
        if (x < remainder) {
          samples_per_buffer_++;
        }
        if (x < offset_) {
          samples_per_buffer_--;
        }
        old_v += samples_per_buffer_;
        // nums_per_shard_ is used to save the current shard's ending index
        nums_per_shard_.push_back(old_v);
      }
    }
    int no_of_categories = static_cast<int>(tasks.categories);
    int64_t total_no = tasks.sample_ids_.size();
    int64_t taking = 0;
    if (sampler_type_ == kCustomTopNSampler) {  // non sharding case constructor #1
      no_of_samples_ = std::min(no_of_samples_, total_no);
      taking = no_of_samples_ - no_of_samples_ % no_of_categories;
    } else if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
      CHECK_FAIL_RETURN_UNEXPECTED_MR(static_cast<int64_t>(indices_.size()) <= total_no,
                                      "Invalid input, indices size: " + std::to_string(indices_.size()) +
                                        " should be less than or equal to database size: " + std::to_string(total_no) +
                                        ".");
    } else {  // constructor TopPercent
      if (numerator_ > 0 && denominator_ > 0 && numerator_ <= denominator_) {
        if (numerator_ == 1 && denominator_ > 1) {  // sharding
          taking = (total_no + denominator_ - 1) / denominator_;
        } else {  // non sharding
          taking = total_no * numerator_ / denominator_;
          taking -= (taking % no_of_categories);
        }
      } else {
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] 'numerator_': " + std::to_string(numerator_) +
                                    " should be positive and less than denominator_: " + std::to_string(denominator_) +
                                    ".");
      }
    }
    return UpdateTasks(tasks, taking);
  }

  std::string env_mindrecord_shard_by_block = common::GetEnv("MS_DEV_MINDRECORD_SHARD_BY_BLOCK");
  (void)transform(env_mindrecord_shard_by_block.begin(), env_mindrecord_shard_by_block.end(),
                  env_mindrecord_shard_by_block.begin(), ::tolower);
  if (env_mindrecord_shard_by_block == "true") {
    return UpdatePartitionWhenSlowMode(tasks);
  } else {
    return UpdatePartitionWhenSlowModeBySlice(tasks);
  }
}

Status ShardSample::SufExecute(ShardTaskList &tasks) {
  if (sampler_type_ == kSubsetRandomSampler) {
    RETURN_IF_NOT_OK_MR((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
