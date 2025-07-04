/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_

#include <deque>
#include <limits>
#include <memory>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class PythonSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param num_samples - the number of samples to draw.  Value of 0 means to sample all of the
  //                      data from the dataset.
  // @param py_sampler_instance - the python instance of the sampler
  // @param int64_t samples_per_tensor - Num of Sampler Ids to fetch via 1 GetNextSample call
  PythonSamplerRT(int64_t num_samples, const py::object &py_sampler_instance,
                  int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~PythonSamplerRT() override;

  // Initialize the sampler.
  // @return Status
  Status InitSampler() override;

  /// \brief Reset for next epoch.
  /// \param[in] failover_reset A boolean to show whether we are resetting the pipeline
  /// \return Status The status code returned
  Status ResetSampler(const bool failover_reset) override;

  // Op calls this to get next Sample that contains all the sampleIds
  // @param TensorRow to be returned to corresponding Dataset Op
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(TensorRow *out) override;

  /// \brief Get batch sizes of this epoch for batch sampler.
  /// \param[in] batch_sizes The list of batch sizes of this epoch.
  /// \return Status The status code.
  Status GetBatchSizes(std::deque<int64_t> *batch_sizes);

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Calculate the number samples.
  /// \param[in] num_rows The input number indices of this sampler.
  /// \return Status The status code.
  int64_t CalculateNumSamples(int64_t num_rows) override;

 private:
  bool need_to_reset_;  // Whether Reset() should be called before calling GetNextSample()

  py::object py_sampler_instance_;  // The handle to the py_sampler python object
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_
