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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_INFO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_INFO_H_

#include <string>

namespace mindspore {
namespace dataset {
enum BatchCtrl : int8_t { kNoCtrl = 0, kEOE = 1, kEOF = 2, kQuit = 3, kWait = 4 };

// Parameters associate with one batch.
// This struct is used for both internal control and python callback.
// This struct is bound to python with read-only access.
struct CBatchInfo {
  CBatchInfo(int64_t ep, int64_t bat, int64_t cur, BatchCtrl ctrl)
      : epoch_num_(ep), batch_num_(bat), total_batch_num_(cur), ctrl_(ctrl) {}
  CBatchInfo(int64_t ep, int64_t bat, int64_t cur) : CBatchInfo(ep, bat, cur, BatchCtrl::kNoCtrl) {}
  CBatchInfo() : CBatchInfo(0, 0, 0, BatchCtrl::kNoCtrl) {}
  explicit CBatchInfo(BatchCtrl ctrl) : CBatchInfo(0, 0, 0, ctrl) {}
  int64_t epoch_num_;        // i-th epoch. i starts from 0
  int64_t batch_num_;        // i-th batch since the start of current epoch. i starts from 0
  int64_t total_batch_num_;  // i-th batch since the start of first epoch. i starts from 0
  BatchCtrl ctrl_;           // No control=0, EOE=1, EOF=2, Quit=3
  const int64_t get_batch_num() const { return batch_num_; }
  const int64_t get_epoch_num() const { return epoch_num_; }

  std::string FlagName() const {
    switch (ctrl_) {
      case BatchCtrl::kNoCtrl:
        return "Data";
      case BatchCtrl::kEOE:
        return "EOE";
      case BatchCtrl::kEOF:
        return "EOF";
      case BatchCtrl::kQuit:
        return "Quit";
      case BatchCtrl::kWait:
        return "Wait";
      default:
        return "Unknown";
    }
  }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_INFO_H_
