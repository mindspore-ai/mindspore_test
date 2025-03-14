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
#include <atomic>
#include <vector>
#include <string>
#include <sstream>
#include <optional>
#include <algorithm>

#include "debug/dump/tensordump_control.h"
#include "utils/distributed_meta.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"

namespace {

template <typename T>
std::string ReplacePlaceholder(const std::string &file_name, const std::string &placeholder, T value) {
  std::string result = file_name;
  const std::string full_placeholder = "{" + placeholder + "}";
  std::ostringstream oss;
  oss << placeholder << value;
  const std::string value_str = oss.str();
  size_t pos = 0;
  while ((pos = result.find(full_placeholder, pos)) != std::string::npos) {
    result.replace(pos, full_placeholder.length(), value_str);
    pos += value_str.length();
  }
  return result;
}

bool EndsWith(const std::string &s, const std::string &sub) {
  if (s.length() < sub.length()) {
    return false;
  }
  return s.rfind(sub) == (s.length() - sub.length()) ? true : false;
}

std::uint32_t GetRankId() {
  std::uint32_t rank_id = 0;
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rank_id = mindspore::DistributedMeta::GetInstance()->global_rank_id();
  }
  return rank_id;
}
}  // namespace
namespace mindspore {

std::string TensorDumpStepManager::TensorNameToArrayName(std::string tensor_path, std::string data_type,
                                                         const int mode) {
  const std::string npy_suffix = ".npy";
  const std::string separator = "_";

  tensor_path = ReplacePlaceholder(tensor_path, "step", GetStep(mode));
  tensor_path = ReplacePlaceholder(tensor_path, "rank", GetRankId());
  std::transform(data_type.begin(), data_type.end(), data_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  std::optional<std::string> parent_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(tensor_path, &parent_path, &file_name);
  if (!parent_path.has_value()) {
    parent_path = ".";
  }
  if (!file_name.has_value()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, failed to extract file name from the arg of 'file', file is "
                  << tensor_path;
    return {};
  }
  std::optional<std::string> realpath = FileUtils::CreateNotExistDirs(*parent_path, true);
  if (EndsWith(*file_name, npy_suffix)) {
    file_name = file_name->substr(0, file_name->length() - npy_suffix.length());
  }
  size_t name_id = TensorDumpStepManager::GetInstance().FetchAddID();
  std::optional<std::string> new_file_name =
    *file_name + separator + data_type + separator + std::to_string(name_id) + npy_suffix;
  std::optional<std::string> new_file_path;
  FileUtils::ConcatDirAndFileName(&realpath, &new_file_name, &new_file_path);
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, mode is " << (mode ? "PYNATIVE" : "GRAPH")
                            << ", dump file path is " << *new_file_path;
  return *new_file_path;
}

void TensorDumpStepManager::SetDumpStep(const std::vector<size_t> &steps) {
  valid_steps_ = {steps.begin(), steps.end()};
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, call set dump step, " << steps << " can be dump.";
}

bool TensorDumpStepManager::NeedDump(const int mode) const {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kPynativeMode || mode == kGraphMode, "Invalid mode");
  auto step = step_.at(mode);
  return valid_steps_.empty() || valid_steps_.find(step) != valid_steps_.end();
}

size_t TensorDumpStepManager::GetStep(const int mode) const {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kPynativeMode || mode == kGraphMode, "Invalid mode");
  return step_.at(mode);
}

void TensorDumpStepManager::UpdateStep(const int mode) {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kPynativeMode || mode == kGraphMode, "Invalid mode");
  step_.at(mode) += 1;
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, mode is " << (mode ? "PYNATIVE" : "GRAPH")
                            << ", after update step, current step is  " << step_.at(mode);
}

size_t TensorDumpStepManager::FetchAddID() { return id_.fetch_add(1, std::memory_order_relaxed); }

std::string TensorDumpStepManager::ProcessFileName(const std::string &filename, const std::string &dtype,
                                                   const int mode) {
  constexpr std::string_view step_flag{"<tensordump-update-step>"};
  if (filename == step_flag) {
    UpdateStep(mode);
    return {};
  }
  if (!NeedDump(mode)) {
    return {};
  }
  return TensorNameToArrayName(filename, dtype, mode);
}

void TensorDumpStepManager::SetAclDumpCallbackReg(void *callbackReg) { aclDumpCallbackReg_ = callbackReg; }
}  // namespace mindspore
