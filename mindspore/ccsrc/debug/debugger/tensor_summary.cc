/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "debug/debugger/tensor_summary.h"
#include <cmath>
#include <algorithm>
#include <future>
#include <limits>
#include <memory>
#include <bitset>
#include <tuple>
#include <type_traits>

#ifdef OFFLINE_DBG_MODE
#include "base/float16.h"
#endif

namespace mindspore {
RangeCountCalculator::RangeCountCalculator()
    : range_start_inclusive(-std::numeric_limits<double>::infinity()),
      range_end_inclusive(std::numeric_limits<double>::infinity()),
      count(0),
      total(0) {}

void RangeCountCalculator::ProcessElement(double element) {
  if (element >= range_start_inclusive && element <= range_end_inclusive) {
    count += 1;
  }
  total += 1;
}

double RangeCountCalculator::GetPercentInRange() const {
  if (total == 0) {
    return 0.0;
  }
  const double factor = 100.0;
  return factor * count / total;
}

AllCloseCalculator::AllCloseCalculator() : atol(1.0e-8), rtol(1.0e-5), result(true) {}

void AllCloseCalculator::ProcessElement(double current, double previous) {
  result = result && (std::abs(current - previous) <= (atol + rtol * std::abs(previous)));
}

bool AllCloseCalculator::IsAllClose() const { return result; }

MeanCalculator::MeanCalculator() : mean(0.0), count(0) {}

void MeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
}

double MeanCalculator::GetMean() const { return mean; }

VarianceAndMeanCalculator::VarianceAndMeanCalculator() : mean(0.0), count(0), m2(0.0) {}

void VarianceAndMeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
  m2 += delta * (value - mean);
}

double VarianceAndMeanCalculator::GetMean() const { return mean; }

double VarianceAndMeanCalculator::GetVariance() const {
  if (count > 1) {
    return m2 / (count - 1);
  }
  return 0.0;
}

double VarianceAndMeanCalculator::GetStandardDeviation() const { return sqrt(GetVariance()); }

void L2Calculator::ProcessElement(double value) { squre_sum += value * value; }

void L2Calculator::ProcessElement(const L2Calculator &other) { this->squre_sum += other.squre_sum; }

double L2Calculator::GetL2Value() const { return std::sqrt(squre_sum); }

template <typename T>
TensorSummary<T>::TensorSummary(const void *current_tensor_ptr, const void *const previous_tensor_ptr,
                                uint64_t num_elements, uint64_t prev_num_elements)
    : current_tensor_ptr_(static_cast<const T *>(current_tensor_ptr)),
      prev_tensor_ptr_(static_cast<const T *>(previous_tensor_ptr)),
      num_elements_(num_elements),
      prev_num_elements_(prev_num_elements),
      min_(std::numeric_limits<double>::max()),
      max_(std::numeric_limits<double>::lowest()),
      avg_(0.0),
      is_bool_(false),
      neg_zero_count_(0),
      pos_zero_count_(0),
      pos_inf_count_(0),
      neg_inf_count_(0),
      inf_count_(0),
      nan_count_(0),
      zero_count_(0),
      epsilon_(1.0e-9),
      mean_sd_cal_enabled_(false) {}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Calculates statistics on chunks of data.
 */
template <typename T>
void TensorSummary<T>::TensorStatistics(DbgDataType dtype_value) {
  if (dtype_value == DT_BOOL) {
    is_bool_ = true;
  }
  const uint64_t default_threads = 32;
  const uint64_t default_elements_per_thread = 10000;

  if (num_elements_ <= default_elements_per_thread) {
    return TensorStatisticsSingleThread();
  }
  uint64_t desired_threads = num_elements_ / default_elements_per_thread;
  uint64_t actual_threads = std::min(desired_threads, default_threads);
  uint64_t actual_elements_per_thread = num_elements_ / actual_threads;

  // Use multithread to calculate statistic on chunks of data
  void *previous_tensor_ptr = nullptr;
  size_t offset = 0;
  std::vector<std::unique_ptr<TensorSummary<T>>> summary_vec;
  std::vector<std::future<void>> summary_future_vec;
  for (uint64_t i = 0; i < actual_threads; i++) {
    uint64_t num_elements_for_thread;
    if (i == actual_threads - 1) {
      num_elements_for_thread = num_elements_ - offset;
    } else {
      num_elements_for_thread = actual_elements_per_thread;
    }
    (void)summary_vec.emplace_back(std::make_unique<TensorSummary<T>>(current_tensor_ptr_ + offset, previous_tensor_ptr,
                                                                      num_elements_for_thread, 0));
    (void)summary_future_vec.emplace_back(
      std::async(std::launch::async, &TensorSummary<T>::TensorStatisticsSingleThread, summary_vec[i].get()));
    offset += num_elements_for_thread;
  }

  // Aggregate results of all chunks
  num_elements_ = 0;  // Let current tensor weight 0 in the aggregation
  for (unsigned int i = 0; i < summary_future_vec.size(); i++) {
    summary_future_vec[i].wait();
    summary_future_vec[i].get();
    auto &cur_summary = *(summary_vec[i]);
    num_elements_ += cur_summary.num_elements_;
    min_ = std::min(min_, cur_summary.min_);
    max_ = std::max(max_, cur_summary.max_);
    double avg_delta = cur_summary.avg_ - avg_;
    avg_ += avg_delta * (cur_summary.num_elements_ / num_elements_);
    neg_zero_count_ += cur_summary.neg_zero_count_;
    pos_zero_count_ += cur_summary.pos_zero_count_;
    neg_inf_count_ += cur_summary.neg_inf_count_;
    pos_inf_count_ += cur_summary.pos_inf_count_;
    inf_count_ += cur_summary.inf_count_;
    nan_count_ += cur_summary.nan_count_;
    zero_count_ += cur_summary.zero_count_;
    l2_calc_.ProcessElement(cur_summary.l2_calc_);
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Process all the elements of the chunked data and calculates the statistics.
 */
template <typename T>
void TensorSummary<T>::TensorStatisticsSingleThread() {
  MeanCalculator mean_calc = MeanCalculator();
  for (size_t i = 0; i < num_elements_; ++i) {
    auto current_value = static_cast<double>(current_tensor_ptr_[i]);
    l2_calc_.ProcessElement(current_value);
    if (std::isnan(current_value)) {
      nan_count_ += 1;
      max_ = current_value;
      min_ = current_value;
      mean_calc.ProcessElement(current_value);
      continue;
    }
    if (std::isinf(current_value)) {
      if (current_value > 0) {
        pos_inf_count_ += 1;
      } else {
        neg_inf_count_ += 1;
      }
    }
    if (current_value == 0.0) {
      zero_count_ += 1;
    }
    // only considering tensor elements with value
    if (std::signbit(current_value) && !(current_value == 0.0)) {
      neg_zero_count_ += 1;
    } else if (!(current_value == 0.0)) {
      pos_zero_count_ += 1;
    }
    max_ = std::max(max_, current_value);
    min_ = std::min(min_, current_value);
    mean_calc.ProcessElement(current_value);
  }
  avg_ = mean_calc.GetMean();
}

template <typename T>
double_t TensorSummary<T>::GetZeroValPercent() const {
  if (num_elements_ == 0) {
    return 0.0;
  }
  const double percentage = 100;
  return (zero_count_ * percentage) / num_elements_;
}

template class TensorSummary<uint8_t>;
template class TensorSummary<int8_t>;
template class TensorSummary<uint16_t>;
template class TensorSummary<int16_t>;
template class TensorSummary<uint32_t>;
template class TensorSummary<int32_t>;
template class TensorSummary<uint64_t>;
template class TensorSummary<int64_t>;
template class TensorSummary<float16>;
template class TensorSummary<bfloat16>;
template class TensorSummary<float>;
template class TensorSummary<double>;
template class TensorSummary<bool>;
}  // namespace mindspore
