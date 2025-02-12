/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/sampler/python_sampler.h"

#include <algorithm>
#include <string>
#include <utility>

#include "minddata/dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {
PythonSamplerRT::PythonSamplerRT(int64_t num_samples, const py::object &py_sampler_instance, int64_t samples_per_tensor)
    : SamplerRT(num_samples, samples_per_tensor), need_to_reset_(false) {
  py::gil_scoped_acquire gil_acquire;
  py_sampler_instance_ = py_sampler_instance;
}

PythonSamplerRT::~PythonSamplerRT() {
  py::gil_scoped_acquire gil_acquire;
  py_sampler_instance_ = py::object();
}

Status PythonSamplerRT::GetBatchSizes(std::deque<int64_t> *batch_sizes) {
  RETURN_UNEXPECTED_IF_NULL(batch_sizes);
  *batch_sizes = std::deque<int64_t>();
  {
    py::gil_scoped_acquire gil_acquire;
    try {
      py::object py_ret = py_sampler_instance_.attr("_get_batch_sizes")();
      auto sizes = py::cast<py::list>(py_ret);
      (void)std::transform(sizes.begin(), sizes.end(), std::back_inserter(*batch_sizes),
                           [](const auto &size) { return py::cast<int64_t>(size); });
    } catch (const py::error_already_set &e) {
      RETURN_STATUS_UNEXPECTED(e.what());
    }
  }
  return Status::OK();
}

Status PythonSamplerRT::GetNextSample(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  if (need_to_reset_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> sample_ids;
    {
      py::gil_scoped_acquire gil_acquire;
      if (Py_IsInitialized() == 0) {
        return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
      }
      try {
        py::object py_ret = py_sampler_instance_.attr("_get_indices")();
        py::array np_sample_ids = py_ret.cast<py::array>();
        std::shared_ptr<Tensor> ids_from_np;
        RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(np_sample_ids, &ids_from_np));  // copy numpy to tensor
        // date type of sample ids got from numpy is not sure on different platforms, so cast it to int64
        CHECK_FAIL_RETURN_UNEXPECTED(
          ids_from_np->type().IsInt(),
          "Python sampler should return index of type integer, but got: " + ids_from_np->type().ToString());
        RETURN_IF_NOT_OK(TypeCast(ids_from_np, &sample_ids, DataType(DataType::DE_INT64)));

        if (HasChildSampler()) {
          for (auto it = sample_ids->begin<int64_t>(); it != sample_ids->end<int64_t>(); ++it) {
            RETURN_IF_NOT_OK(GetAssociatedChildId(it.operator->(), *it));
          }
        }
      } catch (const py::error_already_set &e) {
        return Status(StatusCode::kMDPyFuncException, e.what());
      } catch (const py::cast_error &e) {
        return Status(StatusCode::kMDPyFuncException,
                      "Invalid data, Python sampler iterator should return an integer index, but error raised: " +
                        std::string(e.what()));
      }
    }
    (*out) = {sample_ids};
    need_to_reset_ = true;
  }
  return Status::OK();
}

Status PythonSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0, "[Internal ERROR] num_rows must be greater than 0, but got " + std::to_string(num_rows_));
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
    }
    try {
      py_sampler_instance_.attr("_handshake")(num_rows_, num_samples_);
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException,
                    "[Internal ERROR] python sampler execute _handshake failed: " + std::string(e.what()));
    }
  }

  is_initialized = true;
  return Status::OK();
}

Status PythonSamplerRT::ResetSampler(const bool failover_reset) {
  if (failover_reset) {
    {
      py::gil_scoped_acquire gil_acquire;
      try {
        (void)py_sampler_instance_.attr("_get_indices")();
        (void)py_sampler_instance_.attr("_get_batch_sizes")();
      } catch (const py::error_already_set &e) {
        RETURN_STATUS_UNEXPECTED(e.what());
      }
    }
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(need_to_reset_, "[Internal ERROR] ResetSampler() called early or late.");
  need_to_reset_ = false;
  py::gil_scoped_acquire gil_acquire;
  if (Py_IsInitialized() == 0) {
    return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
  }
  try {
    py_sampler_instance_.attr("reset")();
  } catch (const py::error_already_set &e) {
    return Status(StatusCode::kMDPyFuncException, e.what());
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler(failover_reset));
  }

  return Status::OK();
}

void PythonSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: PythonSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}
}  // namespace dataset
}  // namespace mindspore
