/**
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.h
 *
 * Additional modifications made by Huawei Technologies Co., Ltd in 2023.
 ==============================================================================*/

#ifndef SRC_COMMON_RANGE_SAMPLER_H_
#define SRC_COMMON_RANGE_SAMPLER_H_

#include <vector>
#include <cstdint>
#include <random>
#include "inc/kernel_log.h"

namespace aicpu {
class RangeSampler {
 public:
  explicit RangeSampler(int64_t range) : range_(range) {}
  virtual ~RangeSampler();

  virtual int64_t Sample(CpuKernelContext &ctx) const = 0;

  virtual float Probability(int64_t value) const = 0;

  void SampleBatch(bool unique, const std::vector<int64_t> &batch) const;

  void SampleBatchGetExpectedCount(CpuKernelContext &ctx, bool unique, int64_t seed, std::vector<int64_t> *batch,
                                   std::vector<float> *batch_expected_count, std::vector<int64_t> extras,
                                   std::vector<float> *extras_expected_count) const;

  virtual void SampleBatchGetExpectedCountAvoid(CpuKernelContext &ctx, bool unique, int64_t seed,
                                                std::vector<int64_t> *batch, std::vector<float> *batch_expected_count,
                                                std::vector<int64_t> extras, std::vector<float> *extras_expected_count,
                                                std::vector<int64_t> avoided_values) const;

  int64_t range() { return range_; }

 protected:
  const int64_t range_;
  mutable std::mt19937 rng_;
};

class LogUniformSampler : public RangeSampler {
 public:
  explicit LogUniformSampler(int64_t range);

  ~LogUniformSampler() override {}

  int64_t Sample(CpuKernelContext &ctx) const override;

  float Probability(int64_t value) const override;

 private:
  const double log_range_;
};

class UniformSampler : public RangeSampler {
 public:
  explicit UniformSampler(int64_t range);

  ~UniformSampler() override {}

  float Probability(int64_t value) const override;

  int64_t Sample(CpuKernelContext &ctx) const override;

 private:
  const float inv_range_;
};
}  // namespace aicpu
#endif  // SRC_COMMON_RANGE_SAMPLER_H_
