/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  https://github.com/tensorflow/tensorflow/blob/v2.6.2/tensorflow/core/kernels/image/scale_and_translate_op.cc

  Additional modifications made by Huawei Technologies Co., Ltd in 2022-2025.
*/

#include "kernel/cpu/native/scale_and_translate_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <type_traits>
#include "mindspore/ops/infer/scale_and_translate.h"
#include "mindspore/ops/infer/grad/scale_and_translate_grad.h"
#include "Eigen/Eigen"

namespace mindspore {
namespace kernel {
namespace scale_and_translate_cpu {
namespace {
constexpr size_t kScaleAndTranslateInputsNum = 4;
constexpr size_t kScaleAndTranslateOutputsNum = 1;
constexpr size_t kScaleAndTranslateGradInputsNum = 4;
constexpr size_t kScaleAndTranslateGradOutputsNum = 1;
constexpr float kScaleAndTranslateBlock = 1000.0f;
constexpr size_t i0 = 0;
constexpr size_t i1 = 1;
constexpr size_t i2 = 2;
constexpr size_t i3 = 3;
}  // namespace

bool ScaleAndTranslateCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  input0_dtype_ = inputs[kIndex0]->dtype_id();
  kernel_type_ = GetValue<std::string>(primitive_->GetAttr(ops::kKernelType));
  antialias_ = GetValue<bool>(primitive_->GetAttr(ops::kAntialias));
  switch (input0_dtype_) {
    case kNumberTypeInt8:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int8_t>;
      break;
    case kNumberTypeInt16:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int16_t>;
      break;
    case kNumberTypeInt32:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int32_t>;
      break;
    case kNumberTypeInt64:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int64_t>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<float16>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat64:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<double>;
      break;
    default:
      MS_LOG(ERROR) << "ScaleAndTranslate kernel does not support " << TypeIdToString(input0_dtype_);
      return false;
  }
  return true;
}

bool ScaleAndTranslateGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  kernel_type_ = GetValue<std::string>(primitive_->GetAttr(ops::kKernelType));
  antialias_ = GetValue<bool>(primitive_->GetAttr(ops::kAntialias));
  kernel_func_ = &ScaleAndTranslateGradCpuKernelMod::LaunchKernel<float>;
  return true;
}

template <typename T>
void ScaleAndTranslateCpuKernelMod::GatherRows(int64_t span_size, const int64_t *starts, const float *weights,
                                               const T *image, const int64_t input_height, const int64_t input_width,
                                               const int64_t output_height, const int64_t output_width,
                                               const int64_t channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto task = [span_size, starts, weights, image, input_height, output, in_row_size, out_row_size](int64_t start,
                                                                                                   int64_t end) {
    for (int64_t y = start; y < end; ++y) {
      float *const out_row = output + out_row_size * y;
      std::fill_n(out_row, out_row_size, 0.0f);

      const int64_t row_start = starts[y];
      const float *const weights_y = weights + y * span_size;
      const int64_t max_source_row = std::min(row_start + span_size, input_height);
      const int64_t real_span_size = max_source_row - row_start;
      const T *in_row_ptr = image + in_row_size * row_start;

      if constexpr (std::is_same<T, float>::value) {
        Eigen::Map<Eigen::ArrayXf> out_map(out_row, out_row_size);
        for (int64_t i = 0; i < real_span_size; ++i) {
          const float w = weights_y[i];
          if (w == 0.0f) {
            in_row_ptr += in_row_size;
            continue;
          }
          Eigen::Map<const Eigen::ArrayXf> in_map(in_row_ptr, in_row_size);
          out_map += w * in_map;
          in_row_ptr += in_row_size;
        }
      } else {
        for (int64_t i = 0; i < real_span_size; ++i) {
          const float w = weights_y[i];
          if (w == 0.0f) {
            in_row_ptr += in_row_size;
            continue;
          }
          const T *in_vec = in_row_ptr;
          float *out_vec = out_row;
          const int64_t n = in_row_size;
          const int64_t unroll_step = 4;
          const int64_t n_unrolled = n - (n % unroll_step);
          for (int64_t idx = 0; idx < n_unrolled; idx += unroll_step) {
            out_vec[idx] += w * static_cast<float>(in_vec[idx]);
            out_vec[idx + i1] += w * static_cast<float>(in_vec[idx + i1]);
            out_vec[idx + i2] += w * static_cast<float>(in_vec[idx + i2]);
            out_vec[idx + i3] += w * static_cast<float>(in_vec[idx + i3]);
          }
          for (int64_t idx = n_unrolled; idx < n; ++idx) {
            out_vec[idx] += w * static_cast<float>(in_vec[idx]);
          }
          in_row_ptr += in_row_size;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, output_height, this, &parallel_search_info_);
}

template <typename T>
void ScaleAndTranslateCpuKernelMod::GatherColumns(int64_t span_size, const int64_t *starts, const float *weights,
                                                  const T *image, const int64_t input_height, const int64_t input_width,
                                                  const int64_t output_height, const int64_t output_width,
                                                  const int64_t channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto task = [span_size, starts, weights, image, input_height, input_width, output_width, channels, output,
               in_row_size, out_row_size](int64_t start, int64_t end) {
    for (int64_t y = start; y < end; ++y) {
      const T *input_row_start = image + in_row_size * y;
      float *out_pix = output + out_row_size * y;
      for (int64_t x = 0; x < output_width; ++x, out_pix += channels) {
        const int64_t start_col = starts[x];
        const T *in_pix = input_row_start + start_col * channels;
        const float *weights_start = weights + x * span_size;
        const int64_t max_source_col = std::min(start_col + span_size, input_width);
        const int64_t real_span_size = max_source_col - start_col;

        if constexpr (std::is_same<T, float>::value) {
          Eigen::Map<Eigen::ArrayXf> out_map(out_pix, channels);
          out_map.setZero();
          for (int64_t i = 0; i < real_span_size; ++i) {
            const float w = weights_start[i];
            if (w == 0.0f) {
              in_pix += channels;
              continue;
            }
            Eigen::Map<const Eigen::ArrayXf> in_map(in_pix, channels);
            out_map += w * in_map;
            in_pix += channels;
          }
        } else {
          std::fill_n(out_pix, channels, 0.0f);
          for (int64_t i = 0; i < real_span_size; ++i) {
            const float w = weights_start[i];
            if (w == 0.0f) {
              in_pix += channels;
              continue;
            }
            const T *in_vec = in_pix;
            float *out_vec = out_pix;
            const int64_t n = channels;
            const int64_t unroll_step = 4;
            const int64_t n_unrolled = n - (n % unroll_step);
            for (int64_t idx = 0; idx < n_unrolled; idx += unroll_step) {
              out_vec[idx] += w * static_cast<float>(in_vec[idx]);
              out_vec[idx + i1] += w * static_cast<float>(in_vec[idx + i1]);
              out_vec[idx + i2] += w * static_cast<float>(in_vec[idx + i2]);
              out_vec[idx + i3] += w * static_cast<float>(in_vec[idx + i3]);
            }
            for (int64_t idx = n_unrolled; idx < n; ++idx) {
              out_vec[idx] += w * static_cast<float>(in_vec[idx]);
            }
            in_pix += channels;
          }
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, output_height, this, &parallel_search_info_);
}

template <typename T>
uint32_t ScaleAndTranslateCpuKernelMod::GatherSpans(
  int64_t row_span_size, Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts,
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights, int64_t col_span_size,
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts, Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights,
  typename TTypes<T, dim4>::Tensor images, Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_buffer,
  typename TTypes<float, dim4>::Tensor resized_images) {
  const int64_t batch_size = images.dimension(0);
  const int64_t input_height = images.dimension(1);
  const int64_t input_width = images.dimension(2);
  const int64_t channels = images.dimension(3);
  const int64_t output_height = resized_images.dimension(1);
  const int64_t output_width = resized_images.dimension(2);

  const int64_t input_pix_per_batch = input_width * input_height * channels;
  const int64_t intermediate_pix_per_batch = input_width * output_height * channels;
  const int64_t output_pix_per_batch = output_width * output_height * channels;

  const int64_t *row_start_data = row_starts.data();
  const float *row_weights_data = row_weights.data();
  const int64_t *col_start_data = col_starts.data();
  const float *col_weights_data = col_weights.data();

  const T *image_ptr = images.data();
  float *intermediate_ptr = intermediate_buffer.data();
  float *out_ptr = resized_images.data();

  for (int64_t b = 0; b < batch_size; ++b) {
    GatherRows(row_span_size, row_start_data, row_weights_data, image_ptr, input_height, input_width, output_height,
               input_width, channels, intermediate_ptr);
    GatherColumns(col_span_size, col_start_data, col_weights_data, intermediate_ptr, output_height, input_width,
                  output_height, output_width, channels, out_ptr);
    image_ptr += input_pix_per_batch;
    intermediate_ptr += intermediate_pix_per_batch;
    out_ptr += output_pix_per_batch;
  }
  return true;
}

template <typename Kernel>
void ScaleAndTranslateCpuKernelMod::ComputeSpansCore(const Kernel &kernel, const int64_t output_size,
                                                     const int64_t input_size, const float scale, const float translate,
                                                     bool antialias, Spans *spans) {
  // Constants and validation
  constexpr float kEpsilon = 1e-5f;
  constexpr float kMinWeightThreshold = 1000.0f;

  if (std::abs(scale) <= kEpsilon) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", divisor scale cannot be 0.";
  }

  // Pre-compute constants to avoid repeated calculations
  const float inv_scale = 1.0f / scale;
  const float inv_translate = -inv_scale * translate;
  const float kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
  const float one_over_kernel_scale = 1.0f / kernel_scale;
  const float kernel_radius = kernel.Radius();

  // Calculate span size with bounds checking
  const int64_t calculated_span_size = 2 * FloatToInt(std::ceil(kernel_radius * kernel_scale)) + 1;
  spans->span_size = std::min(calculated_span_size, input_size);

  // Allocate memory for spans
  spans->starts = std::make_shared<Eigen::Tensor<int64_t, dim1>>(output_size);
  spans->weights = std::make_shared<Eigen::Tensor<float, dim1>>(spans->span_size * output_size);

  // Create tensor maps for efficient access
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> weights_vec(spans->weights->data(), spans->weights->dimensions());

  // Initialize weights to zero
  (void)weights_vec.setZero();

  // Lambda function for clamping values within bounds (optimized version)
  auto clamp = [](const auto &low, const auto &high, const auto &value) -> const auto & {
    return (value < low) ? low : (high < value) ? high : value;
  };

  // Process each output position
  for (int64_t x = 0; x < output_size; ++x) {
    // Calculate sampling position
    const float sample_f = (x + 0.5f) * inv_scale + inv_translate;

    // Skip if sampling location is outside the source image
    if (sample_f < 0.0f || sample_f > static_cast<float>(input_size)) {
      starts_vec(x) = 0;
      continue;
    }

    // Calculate span boundaries
    const float span_start_f = sample_f - kernel_radius * kernel_scale - 0.5f;
    const float span_end_f = sample_f + kernel_radius * kernel_scale - 0.5f;

    int64_t span_start = std::ceil(span_start_f);
    int64_t span_end = std::floor(span_end_f) + 1;

    // Clamp span boundaries to valid input range
    span_start = clamp(IntToLong(0), input_size - 1, span_start);
    span_end = clamp(IntToLong(0), input_size, span_end);

    const int64_t this_span_size = span_end - span_start;

    // Validate span size
    if (this_span_size > spans->span_size) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", span size cannot be larger than " << spans->span_size
                        << ", but got " << this_span_size << ".";
    }

    // Skip if span is empty
    if (this_span_size <= 0) {
      starts_vec(x) = span_start;
      continue;
    }

    // Calculate kernel weights
    std::vector<float> temp_weights;
    temp_weights.reserve(this_span_size);
    temp_weights.clear();
    float total_weight_sum = 0.0f;

    // Pre-calculate weight sum to avoid division by zero
    for (int64_t source = span_start; source < span_end; ++source) {
      const float kernel_pos = LongToFloat(source) + 0.5f - sample_f;
      const float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
      temp_weights.push_back(weight);
      total_weight_sum += weight;
    }

    // Normalize weights only if sum is significant
    if (std::abs(total_weight_sum) >= kMinWeightThreshold * std::numeric_limits<float>::min()) {
      const float one_over_total_weight_sum = 1.0f / total_weight_sum;
      const int64_t out_index = spans->span_size * x;

      // Apply normalization and store weights (vectorized approach)
      for (size_t i = 0; i < temp_weights.size(); ++i) {
        weights_vec(out_index + i) = temp_weights[i] * one_over_total_weight_sum;
      }
    }

    // Store span start position
    starts_vec(x) = span_start;
  }
}

void ScaleAndTranslateGradCpuKernelMod::ComputeGradSpansCore(const Spans *spans, const int64_t forward_output_size,
                                                             const int64_t forward_input_size, Spans *grad_spans) {
  // Use more efficient data structure: reserve capacity and avoid repeated allocations
  struct GradComponent {
    int64_t index;
    float weight;

    // Custom comparator for efficient sorting
    bool operator<(const GradComponent &other) const { return index < other.index; }
  };

  // Pre-allocate vectors with estimated capacity to reduce reallocations
  std::vector<std::vector<GradComponent>> grad_components(forward_input_size);
  const size_t estimated_components_per_input =
    std::min(static_cast<size_t>(spans->span_size), static_cast<size_t>(forward_output_size));

  for (auto &component_list : grad_components) {
    component_list.reserve(estimated_components_per_input);
  }

  // Create tensor maps for efficient access
  const Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  const Eigen::TensorMap<Eigen::Tensor<float, dim1>> weights_vec(spans->weights->data(), spans->weights->dimensions());

  // First phase: collect gradient components with improved parallelization
  auto collect_grad_components = [spans, &starts_vec, &weights_vec, &grad_components, forward_input_size](int64_t start,
                                                                                                          int64_t end) {
    for (int64_t output_index = start; output_index < end; ++output_index) {
      const int64_t input_start = starts_vec(output_index);
      const int64_t span_size = spans->span_size;

      // Process each span element
      for (int64_t j = 0; j < span_size; ++j) {
        const int64_t input_index = input_start + j;
        if (input_index >= forward_input_size) {
          continue;  // Skip out-of-bounds indices
        }

        const float weight = weights_vec(output_index * span_size + j);
        // Use direct comparison for better performance (avoid function call overhead)
        if (std::abs(weight) > std::numeric_limits<float>::epsilon()) {
          grad_components[input_index].emplace_back(GradComponent{output_index, weight});
        }
      }
    }
  };

  // Optimize parallelization strategy based on problem size
  const int64_t parallel_threshold = kScaleAndTranslateBlock;
  if (forward_output_size < parallel_threshold) {
    // For small problems, use simple parallel launch
    ParallelLaunch(collect_grad_components, forward_output_size, parallel_threshold);
  } else {
    // For larger problems, use auto-search for optimal chunk size
    ParallelLaunchAutoSearch(collect_grad_components, forward_output_size, this, &parallel_search_info_);
  }

  // Second phase: calculate optimal span size and prepare output structures
  int64_t max_span_size = 0;
  std::vector<int64_t> valid_input_indices;
  valid_input_indices.reserve(forward_input_size);

  // Find valid inputs and calculate max span size in a single pass
  for (int64_t i = 0; i < forward_input_size; ++i) {
    if (!grad_components[i].empty()) {
      valid_input_indices.push_back(i);

      // Sort components for efficient span calculation
      std::sort(grad_components[i].begin(), grad_components[i].end());

      // Calculate span size for this input
      const int64_t span_size = grad_components[i].back().index - grad_components[i].front().index + 1;
      max_span_size = std::max(max_span_size, span_size);
    }
  }

  // Set output span size
  grad_spans->span_size = max_span_size;

  // Allocate output tensors with proper dimensions
  grad_spans->starts = std::make_shared<Eigen::Tensor<int64_t, dim1>>(forward_input_size);
  grad_spans->weights = std::make_shared<Eigen::Tensor<float, dim1>>(max_span_size * forward_input_size);

  // Create tensor maps for output
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> grad_starts_vec(grad_spans->starts->data(),
                                                                 grad_spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> grad_weights_vec(grad_spans->weights->data(),
                                                                grad_spans->weights->dimensions());

  // Initialize weights to zero efficiently
  (void)grad_weights_vec.setZero();

  // Third phase: populate output tensors with optimized parallel processing
  auto populate_output_tensors = [&valid_input_indices, &grad_components, &grad_starts_vec, &grad_weights_vec,
                                  max_span_size](int64_t start, int64_t end) {
    for (int64_t idx = start; idx < end; ++idx) {
      const int64_t input_index = valid_input_indices[idx];
      const auto &component_list = grad_components[input_index];

      if (!component_list.empty()) {
        // Set start position
        const int64_t start_span = component_list.front().index;
        grad_starts_vec(input_index) = start_span;

        // Accumulate weights efficiently
        const int64_t base_offset = input_index * max_span_size;
        for (const auto &component : component_list) {
          const int64_t weight_offset = base_offset + (component.index - start_span);
          grad_weights_vec(weight_offset) += component.weight;
        }
      } else {
        // Set default values for empty inputs
        grad_starts_vec(input_index) = 0;
      }
    }
  };

  // Use parallel processing for output population
  if (valid_input_indices.size() < parallel_threshold) {
    ParallelLaunch(populate_output_tensors, valid_input_indices.size(), parallel_threshold);
  } else {
    ParallelLaunchAutoSearch(populate_output_tensors, valid_input_indices.size(), this, &parallel_search_info_);
  }
}

bool ScaleAndTranslateCpuKernelMod::ComputeSpans(const KernelType kernel_type, const int64_t output_size,
                                                 const int64_t input_size, const float scale, const float translate,
                                                 const bool antialias, Spans *spans, const std::string kernel_name) {
  switch (kernel_type) {
    case Lanczos1: {
      ComputeSpansCore(CreateLanczos1Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Lanczos3: {
      ComputeSpansCore(CreateLanczos3Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Lanczos5: {
      ComputeSpansCore(CreateLanczos5Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Gaussian: {
      ComputeSpansCore(CreateGaussianKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Box: {
      ComputeSpansCore(CreateBoxKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Triangle: {
      ComputeSpansCore(CreateTriangleKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case KeysCubic: {
      ComputeSpansCore(CreateKeysCubicKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case MitchellCubic: {
      ComputeSpansCore(CreateMitchellCubicKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "For " << kernel_name << ", kernel_type kernel data type [" << kernel_type
                        << "] not support.";
      return false;
  }
  return true;
}

bool ScaleAndTranslateGradCpuKernelMod::ComputeGradSpans(const KernelType kernel_type,
                                                         const int64_t forward_output_size,
                                                         const int64_t forward_input_size, const float scale,
                                                         const float translate, const bool antialias, Spans *grad_spans,
                                                         const std::string kernel_name) {
  Spans spans;
  ScaleAndTranslateCpuKernelMod scale_and_translate_mod;
  (void)scale_and_translate_mod.ComputeSpans(kernel_type, forward_output_size, forward_input_size, scale, translate,
                                             antialias, &spans, kernel_name);
  ComputeGradSpansCore(&spans, forward_output_size, forward_input_size, grad_spans);
  return true;
}

template <typename T>
bool ScaleAndTranslateCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                 const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScaleAndTranslateInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScaleAndTranslateOutputsNum, kernel_name_);
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto input_size = GetDeviceAddress<int32_t>(inputs, kIndex1);
  auto input_scale = GetDeviceAddress<float>(inputs, kIndex2);
  auto input_translation = GetDeviceAddress<float>(inputs, kIndex3);
  auto output = GetDeviceAddress<float>(outputs, kIndex0);
  KernelType sampling_kernel_type = GetSamplingKernelType(kernel_type_);
  const int64_t output_height = IntToLong(input_size[0]);
  const int64_t output_width = IntToLong(input_size[1]);
  const int64_t batch_size = input0_shape_[0];
  const int64_t input_height = input0_shape_[1];
  const int64_t input_width = input0_shape_[2];
  const int64_t channels = input0_shape_[3];
  if (output_height <= 0 || output_width <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", output_height and output_width must be positive, but got "
                      << "output_height: " << output_height << " and output_width: " << output_width << ".";
  }
  if (channels <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", image_channel must have at least one, but got image_channel: " << channels << ".";
  }
  if (input_height <= 0 || input_width <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", input_height and input_width must be of non-zero size, but got "
                      << "input_height: " << input_height << " and input_width: " << input_width << ".";
  }
  float row_scale = input_scale[0];
  float col_scale = input_scale[1];
  if (row_scale <= 0 || col_scale <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", row_scale and col_scale must be greater than zero, but got "
                      << "row_scale: " << row_scale << " and col_scale: " << col_scale << ".";
  }
  float row_translation = input_translation[0];
  float col_translation = input_translation[1];
  EigenTensor inputTensor(input0_shape_, input);
  EigenTensor outputTensor(output_shape_, output);
  typename TTypes<T, dim4>::Tensor image_data(inputTensor.tensor<T, dim4>());

  typename TTypes<float, dim4>::Tensor output_data(outputTensor.tensor<float, dim4>());
  Spans col_spans;
  (void)ComputeSpans(sampling_kernel_type, output_width, input_width, col_scale, col_translation, antialias_,
                     &col_spans, kernel_name_);

  Spans row_spans;
  (void)ComputeSpans(sampling_kernel_type, output_height, input_height, row_scale, row_translation, antialias_,
                     &row_spans, kernel_name_);

  Eigen::Tensor<float, dim4> intermediate_tensor_middle(batch_size, output_height, input_width, channels);
  Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_data(intermediate_tensor_middle.data(),
                                                                 intermediate_tensor_middle.dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());
  GatherSpans<T>(row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts, col_weights, image_data,
                 intermediate_data, output_data);
  return true;
}

template <typename T>
bool ScaleAndTranslateGradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                     const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScaleAndTranslateGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScaleAndTranslateGradOutputsNum, kernel_name_);
  auto input = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto input_scale = reinterpret_cast<float *>(inputs[2]->device_ptr());
  auto input_translation = reinterpret_cast<float *>(inputs[3]->device_ptr());
  auto output = reinterpret_cast<float *>(outputs[0]->device_ptr());
  KernelType sampling_kernel_type = GetSamplingKernelType(kernel_type_);

  const int64_t batch_size = input0_shape_[0];
  const int64_t forward_input_height = input1_shape_[1];
  const int64_t forward_input_width = input1_shape_[2];
  const int64_t channels = input0_shape_[3];
  float row_scale = input_scale[0];
  float col_scale = input_scale[1];
  if (row_scale <= 0 || col_scale <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", row_scale and col_scale must be greater than zero, but got "
                      << "row_scale: " << row_scale << " and col_scale: " << col_scale << ".";
    return false;
  }
  float row_translation = input_translation[0];
  float col_translation = input_translation[1];
  EigenTensor inputTensor(input0_shape_, input);
  // output shape should be {batch_size, forward_input_height,forward_input_width, channels};
  EigenTensor outputTensor(output_shape_, output);
  TTypes<float, dim4>::Tensor input_grad(inputTensor.tensor<float, dim4>());
  typename TTypes<T, dim4>::Tensor output_grad(outputTensor.tensor<T, dim4>());
  const int64_t forward_output_height = input_grad.dimension(1);
  const int64_t forward_output_width = input_grad.dimension(2);

  Spans col_spans;
  (void)ComputeGradSpans(sampling_kernel_type, forward_output_width, forward_input_width, col_scale, col_translation,
                         antialias_, &col_spans, kernel_name_);
  Spans row_spans;
  (void)ComputeGradSpans(sampling_kernel_type, forward_output_height, forward_input_height, row_scale, row_translation,
                         antialias_, &row_spans, kernel_name_);

  Eigen::Tensor<float, dim4> intermediate_tensor_middle(batch_size, forward_input_height, forward_output_width,
                                                        channels);
  Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_data(intermediate_tensor_middle.data(),
                                                                 intermediate_tensor_middle.dimensions());

  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());

  ScaleAndTranslateCpuKernelMod scale_and_translate_mod;
  scale_and_translate_mod.GatherSpans<T>(row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts,
                                         col_weights, input_grad, intermediate_data, output_grad);
  return true;
}

int ScaleAndTranslateCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  input1_shape_ = inputs[kIndex1]->GetShapeVector();
  input2_shape_ = inputs[kIndex2]->GetShapeVector();
  input3_shape_ = inputs[kIndex3]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  return 0;
}

int ScaleAndTranslateGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  input1_shape_ = inputs[kIndex1]->GetShapeVector();
  input2_shape_ = inputs[kIndex2]->GetShapeVector();
  input3_shape_ = inputs[kIndex3]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  return 0;
}

std::vector<KernelAttr> ScaleAndTranslateCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeInt8)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt16)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

std::vector<KernelAttr> ScaleAndTranslateGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScaleAndTranslate, ScaleAndTranslateCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScaleAndTranslateGrad, ScaleAndTranslateGradCpuKernelMod);
}  // namespace scale_and_translate_cpu
}  // namespace kernel
}  // namespace mindspore
