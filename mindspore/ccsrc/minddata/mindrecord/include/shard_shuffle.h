/**
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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_

#include <random>
#include <string>

#include "minddata/mindrecord/include/shard_operator.h"

namespace mindspore {
namespace mindrecord {
class MINDRECORD_API ShardShuffle : public ShardOperator {
 public:
  explicit ShardShuffle(uint32_t seed = 0, ShuffleType shuffle_type = kShuffleCategory);

  ShardShuffle(uint32_t seed, int64_t no_of_samples, bool replacement, bool reshuffle_each_epoch,
               ShuffleType shuffle_type = kShuffleSample,
               dataset::ShuffleMode shuffle_mode = dataset::ShuffleMode::kGlobal);

  ~ShardShuffle() override{};

  std::string Name() override { return "ShardShuffle"; }

  Status Execute(ShardTaskList &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  // Private helper function
  Status CategoryShuffle(ShardTaskList &tasks);  // NOLINT

  // Keep the file sequence the same but shuffle the data within each file
  Status ShuffleInfile(ShardTaskList &tasks);  // NOLINT

  // Shuffle the file sequence but keep the order of data within each file
  Status ShuffleFiles(ShardTaskList &tasks);  // NOLINT

  uint32_t shuffle_seed_;
  int64_t no_of_samples_;
  bool replacement_;
  bool reshuffle_each_epoch_;
  ShuffleType shuffle_type_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_
