/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "debug/debug_services.h"
#include <dirent.h>
#include <algorithm>
#include <functional>
#include <fstream>
#include <future>
#include <thread>
#include <iterator>
#include <map>
#include <numeric>
#include <limits>
#include <unordered_set>
#include <utility>
#include <regex>
#include <iomanip>
#include "openssl/md5.h"
#include "include/common/debug/common.h"
#include "include/backend/debug/debugger/debugger.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/utils/anfalgo.h"
#include "debug/utils.h"
#include "nlohmann/json.hpp"
#include "debug/debugger/tensor_summary.h"
#include "utils/file_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/ms_utils.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"

namespace mindspore {
namespace {
constexpr int md5_bit_wide = 2;
constexpr int md5_len = 32;
}  // namespace

void openssl_md5(char *input, char *output, int64_t len) {
  unsigned char digest[MD5_DIGEST_LENGTH];
  MD5(reinterpret_cast<unsigned char *>(input), len, reinterpret_cast<unsigned char *>(digest));
  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    int rest_len = md5_len + 1 - i * md5_bit_wide;
    auto ret =
      snprintf_s(&output[i * md5_bit_wide], rest_len, md5_bit_wide, "%02x", static_cast<unsigned int>(digest[i]));
    if (ret < 0) {
      MS_LOG(ERROR) << "snprintf_s encountered an error when record md5, which may lead to incorrect MD5 value in the "
                       "statistic.csv file.";
    } else if (ret >= rest_len) {
      MS_LOG(ERROR) << "snprintf_s output is truncated when record md5, which may lead to incorrect MD5 value in the "
                       "statistic.csv file.";
    }
  }
}

DebugServices::DebugServices() { tensor_loader_ = std::make_shared<TensorLoader>(); }

DebugServices::DebugServices(const DebugServices &other) { tensor_loader_ = other.tensor_loader_; }

DebugServices &DebugServices::operator=(const DebugServices &other) {
  if (this != &other) {
    tensor_loader_ = other.tensor_loader_;
  }
  return *this;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns a tensor summary unique pointer based on the given tensor_dtype, returns nullptr if the type is
 * not supported.
 */
std::unique_ptr<ITensorSummary> GetSummaryPtr(const std::shared_ptr<TensorData> &tensor,
                                              const void *const previous_tensor_ptr, uint64_t num_elements,
                                              uint64_t prev_num_elements, int tensor_dtype) {
  MS_EXCEPTION_IF_NULL(tensor);
  switch (tensor_dtype) {
    case DbgDataType::DT_UINT8: {
      return std::make_unique<TensorSummary<uint8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_INT8: {
      return std::make_unique<TensorSummary<int8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                     prev_num_elements);
    }
    case DbgDataType::DT_UINT16: {
      return std::make_unique<TensorSummary<uint16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT16: {
      return std::make_unique<TensorSummary<int16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_UINT32: {
      return std::make_unique<TensorSummary<uint32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT32:
    case DbgDataType::DT_BASE_INT: {
      return std::make_unique<TensorSummary<int32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_UINT64: {
      return std::make_unique<TensorSummary<uint64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT64: {
      return std::make_unique<TensorSummary<int64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_FLOAT16: {
      return std::make_unique<TensorSummary<float16>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_BFLOAT16: {
      return std::make_unique<TensorSummary<bfloat16>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_FLOAT32:
    case DbgDataType::DT_BASE_FLOAT: {
      return std::make_unique<TensorSummary<float>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                    prev_num_elements);
    }
    case DbgDataType::DT_FLOAT64: {
      return std::make_unique<TensorSummary<double>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                     prev_num_elements);
    }
    case DbgDataType::DT_BOOL: {
      return std::make_unique<TensorSummary<bool>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                   prev_num_elements);
    }
    default:
      MS_LOG(INFO) << "Unsupported tensor type";
      // return a null pointer
      return std::unique_ptr<TensorSummary<int32_t>>{};
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns TensorStat for the given tensor based on the base_summary_ptr.
 */
DebugServices::TensorStat DebugServices::GetTensorStatistics(const std::shared_ptr<TensorData> &tensor) {
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "Tensor is nullptr, returning empty tensor statistics.";
    TensorStat empty_tensor_stat_data;
    return empty_tensor_stat_data;
  }
  std::unique_ptr<ITensorSummary> base_summary_ptr;
  void *previous_tensor_ptr = nullptr;
  base_summary_ptr = GetSummaryPtr(tensor, previous_tensor_ptr, tensor->GetNumElements(), 0, tensor->GetType());
  if (base_summary_ptr == nullptr) {
    MS_LOG(WARNING) << "base_summary_ptr is nullptr, returning empty tensor statistics.";
    TensorStat empty_tensor_stat_data;
    return empty_tensor_stat_data;
  }
  std::string md5 = "";
  MSLogTime msTime;
  auto statistic_category = DumpJsonParser::GetInstance().statistic_category();
  if (std::find(statistic_category.begin(), statistic_category.end(), "md5") != statistic_category.end()) {
    msTime.Start();
    char md5str[33];
    auto ret = memset_s(md5str, sizeof(md5str), '\0', sizeof(md5str));
    if (ret != EOK) {
      MS_LOG(ERROR) << "Failed to call memset_s, skip record MD5.";
    } else {
      openssl_md5(const_cast<char *>(tensor->GetDataPtr()), md5str, tensor->GetByteSize());
      md5 = std::string(md5str);
    }
    msTime.End();
    MS_LOG(DEBUG) << "Calc md5 costs time : " << msTime.GetRunTimeUS() << " microseconds.";
  }
  msTime.Start();
  base_summary_ptr->TensorStatistics(tensor->GetType());
  msTime.End();
  MS_LOG(DEBUG) << "Calc statistic costs time : " << msTime.GetRunTimeUS() << " microseconds.";
  TensorStat tensor_stat_data(
    tensor->GetByteSize(), tensor->GetType(), tensor->GetShape(), base_summary_ptr->is_bool(),
    base_summary_ptr->max_value(), base_summary_ptr->min_value(), base_summary_ptr->avg_value(),
    base_summary_ptr->count(), base_summary_ptr->neg_zero_count(), base_summary_ptr->pos_zero_count(),
    base_summary_ptr->nan_count(), base_summary_ptr->neg_inf_count(), base_summary_ptr->pos_inf_count(),
    base_summary_ptr->zero_count(), base_summary_ptr->l2_value(), md5);

  return tensor_stat_data;
}

std::shared_ptr<TensorData> DebugServices::GetTensor(const std::string &tensor_name) const {
  return tensor_loader_->GetTensor(tensor_name);
}

void DebugServices::EmptyCurrentTensor() { tensor_loader_->EmptyCurrentTensor(); }

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Description: Dump tensor from tensor loader to file.
 */
bool DebugServices::DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const {
  return tensor_loader_->DumpTensorToFile(filepath, tensor_name, slot);
}

bool DebugServices::LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
  return tensor_loader_->LoadNewTensor(tensor, keep_prev);
}

void DebugServices::ResetLoadedTensors() {
  MS_LOG(INFO) << "Resetting loaded tensors";
  tensor_loader_->EmptyCurrentTensor();
}

bool DebugServices::TensorExistsInCurrent(const std::string &tensor_name) {
  return tensor_loader_->TensorExistsInCurrent(tensor_name);
}

}  // namespace mindspore
