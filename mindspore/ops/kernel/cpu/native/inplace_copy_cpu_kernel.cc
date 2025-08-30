/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/cpu/native/inplace_copy_cpu_kernel.h"

#include <memory>
#include <complex>
#include <typeinfo>
#include <algorithm>

#include "kernel/cpu/utils/cpu_utils.h"
#include "kernel/cpu/nnacl/errorcode.h"
#include "kernel/cpu/nnacl/base/broadcast_to.h"

namespace mindspore {
namespace kernel {
namespace inplace_copy_cpu {
namespace {
enum InplaceCopyMode : int {
  NoCastAndNoBroadCast,
  CastAndNoBroadCast,
  NoCastAndBroadCast,
  CastAndBroadCast,
};

void InplaceCopyParallelLaunch(const CTask &task, size_t data_size) {
  constexpr size_t kGrainSize = 32768;
  auto kernel_thread_num = SizeToLong(GetActorMgrInnerThreadPool()->GetKernelThreadNum());
  auto block_size = LongToSize((SizeToLong(data_size) + kernel_thread_num - 1) / kernel_thread_num);
  block_size = std::max(block_size, kGrainSize);
  ParallelLaunch(task, data_size, SizeToFloat(block_size));
}

template <typename T>
void UncontinugousBroadCastTo(T *input, T *output, const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &output_shape) {
  int status = static_cast<int>(NNACL_OK);
  BroadcastShapeInfo shape_info;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    shape_info.input_shape_[i] = LongToInt(input_shape[i]);
  }
  for (size_t i = 0; i < output_shape.size(); ++i) {
    shape_info.output_shape_[i] = LongToInt(output_shape[i]);
  }
  shape_info.input_shape_size_ = SizeToInt(input_shape.size());
  shape_info.output_shape_size_ = SizeToInt(output_shape.size());

  if constexpr (std::is_same_v<T, bool>) {
    status = BroadcastToSize8(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    status = BroadcastToSize8(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    status = BroadcastToSize16(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    status = BroadcastToSize32(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    status = BroadcastToSize64(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    status = BroadcastToSize8(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    status = BroadcastToSize16(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    status = BroadcastToSize32(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    status = BroadcastToSize64(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, float16>) {
    status = BroadcastToSize16(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, bfloat16>) {
    status = BroadcastToSize16(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, float>) {
    status = BroadcastToSize32(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, double>) {
    status = BroadcastToSize64(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    status = BroadcastToSize64(input, &shape_info, output);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    status = BroadcastToSize128(input, &shape_info, output);
  } else {
    MS_LOG(EXCEPTION) << "For InplaceCopy, broadcast got a unsupported data type";
  }

  if (status != static_cast<int>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "For InplaceCopy, failed to broadcast 'input' shape:" << input_shape
                      << " to 'output' shape: " << output_shape << ". Error code: " << status;
  }
}

template <typename T>
void ContinugousBroadCastTo(T *input, T *output, int64_t broadcast_dim, size_t block_size,
                            const std::vector<int64_t> &input_shape, const std::vector<int64_t> &output_shape) {
  std::vector<size_t> input_strides(input_shape.size(), 1);
  std::vector<size_t> output_strides(output_shape.size(), 1);
  std::vector<std::pair<size_t, size_t>> copy_offsets;
  int64_t dim_max = SizeToLong(output_shape.size()) - 1;
  for (int64_t i = dim_max - 1; i >= 0; --i) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
  }
  size_t num_blocks = 1;
  for (int64_t i = 0; i <= broadcast_dim; ++i) {
    num_blocks *= output_shape[i];
  }
  for (size_t i = 0; i < num_blocks; ++i) {
    size_t output_offset = i * output_strides[broadcast_dim];
    size_t input_offset = 0;
    size_t temp = i;
    for (int64_t j = broadcast_dim; j >= 0; --j) {
      size_t dim_index = temp % output_shape[j];
      if (input_shape[j] != 1) {
        input_offset += dim_index * input_strides[j];
      }
      temp /= output_shape[j];
    }
    copy_offsets.emplace_back(std::pair<size_t, size_t>(input_offset, output_offset));
  }

  constexpr size_t kGrainSize = 32768;
  constexpr size_t kParallelDataLenThreshold = kGrainSize * sizeof(T);
  size_t copy_size = block_size * sizeof(T);
  if (copy_offsets.size() * copy_size > kParallelDataLenThreshold) {
    auto copy_task = [input, output, &copy_offsets, copy_size](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        size_t remain_size = copy_size;
        auto dst_ptr = static_cast<char *>(static_cast<void *>(output + copy_offsets[i].second));
        auto src_ptr = static_cast<char *>(static_cast<void *>(input + copy_offsets[i].first));
        while (remain_size > SECUREC_MEM_MAX_LEN) {
          auto ret = memcpy_s(dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
          if (ret != EOK) {
            MS_LOG(EXCEPTION) << "For InplaceCopy, memcpy_s error. Error no: " << ret << ", dst: " << dst_ptr
                              << ", src: " << src_ptr << ", data_size: " << SECUREC_MEM_MAX_LEN;
          }
          remain_size -= SECUREC_MEM_MAX_LEN;
          dst_ptr += SECUREC_MEM_MAX_LEN;
          src_ptr += SECUREC_MEM_MAX_LEN;
        }
        if (remain_size != 0U) {
          auto ret = memcpy_s(dst_ptr, remain_size, src_ptr, remain_size);
          if (ret != EOK) {
            MS_LOG(EXCEPTION) << "For InplaceCopy, memcpy_s error. Error no: " << ret << ", dst: " << dst_ptr
                              << ", src: " << src_ptr << ", data_size: " << remain_size;
          }
        }
      }
    };
    InplaceCopyParallelLaunch(copy_task, copy_offsets.size());
  } else {
    for (size_t i = 0; i < copy_offsets.size(); ++i) {
      auto ret = memcpy_s(output + copy_offsets[i].second, copy_size, input + copy_offsets[i].first, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "ContinugousBroadCastTo memcpy_s error. Error no: " << ret << ".";
      }
    }
  }
}
}  // namespace

bool InplaceCopyCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int InplaceCopyCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  self_shape_ = inputs[kIndex0]->GetShapeVector();
  self_dtype_ = inputs[kIndex0]->dtype_id();

  value_shape_ = inputs[kIndex1]->GetShapeVector();
  value_dtype_ = inputs[kIndex1]->dtype_id();

  mode_ = static_cast<int>(self_dtype_ != value_dtype_) + (static_cast<int>(self_shape_ != value_shape_) << 1);
  is_empty_ = std::any_of(self_shape_.begin(), self_shape_.end(), [](auto dim) { return dim == 0; });
  if (!is_empty_ && self_shape_ != value_shape_) {
    size_t value_rank = value_shape_.size();
    size_t self_rank = self_shape_.size();
    if (self_rank < value_rank) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the rank of 'value' must be less than 'variable', but got the dimension of 'value': "
                        << value_shape_ << " and 'variable': " << self_shape_ << ".";
    }
    if (self_rank > MAX_SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the rank of 'variable' must be less than 8, but got the dimension of 'variable': "
                        << self_shape_ << ".";
    }
    size_t offset = self_rank - value_rank;
    for (size_t i = 0; i < value_rank; ++i) {
      if (value_shape_[i] != self_shape_[i + offset] && value_shape_[i] != 1) {
        MS_LOG(EXCEPTION)
          << "For '" << kernel_name_ << "', when the " << i
          << "'th, the shape of 'value' must be 1 and equal to the shape of 'variable', but got the shape of 'value': "
          << value_shape_ << ", and the shape of 'variable': " << self_shape_;
      }
    }
  }

  return KRET_OK;
}

template <typename S, typename T>
bool InplaceCopyCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *self = GetDeviceAddress<S>(inputs, kIndex0);
  auto *value = GetDeviceAddress<T>(inputs, kIndex1);
  auto *output = GetDeviceAddress<S>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(self);
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(output);

  if (is_empty_) {
    return true;
  }

  switch (static_cast<InplaceCopyMode>(mode_)) {
    case NoCastAndNoBroadCast: {
      auto *new_value = GetDeviceAddress<S>(inputs, kIndex1);
      MS_EXCEPTION_IF_NULL(new_value);
      InplaceCopySameDtypeSameShape(new_value, self, inputs[kIndex1]->size(), inputs[kIndex0]->size());
      break;
    }
    case CastAndNoBroadCast:
      Cast(value, self, inputs[kIndex0]->size() / sizeof(S));
      break;
    case NoCastAndBroadCast: {
      auto *new_value = GetDeviceAddress<S>(inputs, kIndex1);
      MS_EXCEPTION_IF_NULL(new_value);
      InplaceCopyBroadcastTo(new_value, self, value_shape_, self_shape_);
      break;
    }
    case CastAndBroadCast: {
      S *cast_temp = new S[inputs[kIndex1]->size() / sizeof(T)];
      Cast(value, cast_temp, inputs[kIndex1]->size() / sizeof(T));
      InplaceCopyBroadcastTo(cast_temp, self, value_shape_, self_shape_);
      delete[] cast_temp;
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", mode is invalid, but got " << mode_;
  }
  // copy self to output if they have different addresses.
  if (output != self) {
    InplaceCopySameDtypeSameShape(self, output, inputs[kIndex0]->size(), outputs[kIndex0]->size());
  }
  return true;
}

template <typename T>
void InplaceCopyCpuKernelMod::InplaceCopySameDtypeSameShape(T *input, T *output, size_t input_size,
                                                            size_t output_size) {
  if (output_size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", an unexpected kernel has been launched, but got dst(type: " << typeid(T).name()
                      << ", mem_size: " << output_size << "), src(type: " << typeid(T).name()
                      << ", mem_size: " << input_size << ").";
  }

  constexpr size_t kDataLenThreshold = 32768 * sizeof(T);
  auto data_size = output_size;
  if (data_size <= kDataLenThreshold) {
    auto ret = memcpy_s(output, data_size, input, data_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret << ", dst: " << output
                        << ", src: " << input << ", data_size: " << data_size;
    }
  } else {
    auto inplace_copy_task = [this, input, output](size_t start, size_t end) {
      size_t remain_size = LongToSize((SizeToLong(end) - SizeToLong(start))) * sizeof(T);
      auto dst_ptr = static_cast<char *>(static_cast<void *>(output + start));
      auto src_ptr = static_cast<char *>(static_cast<void *>(input + start));
      while (remain_size > SECUREC_MEM_MAX_LEN) {
        auto ret = memcpy_s(dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For InplaceCopy, memcpy_s error. Error no: " << ret << ", dst: " << dst_ptr
                            << ", src: " << src_ptr << ", data_size: " << SECUREC_MEM_MAX_LEN;
        }
        remain_size -= SECUREC_MEM_MAX_LEN;
        dst_ptr += SECUREC_MEM_MAX_LEN;
        src_ptr += SECUREC_MEM_MAX_LEN;
      }
      if (remain_size != 0U) {
        auto ret = memcpy_s(dst_ptr, remain_size, src_ptr, remain_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For InplaceCopy, memcpy_s error. Error no: " << ret << ", dst: " << dst_ptr
                            << ", src: " << src_ptr << ", data_size: " << remain_size;
        }
      }
    };
    InplaceCopyParallelLaunch(inplace_copy_task, data_size / sizeof(T));
  }
}

template <typename T>
void InplaceCopyCpuKernelMod::InplaceCopyBroadcastTo(T *input, T *output, const std::vector<int64_t> &input_shape,
                                                     const std::vector<int64_t> &output_shape) {
  auto pad_input_shape = input_shape;
  auto input_rank = input_shape.size();
  auto output_rank = output_shape.size();
  if (input_rank != output_rank) {
    std::vector<int64_t> padding_dims(SizeToLong(output_rank) - SizeToLong(input_rank), 1);
    pad_input_shape.insert(pad_input_shape.begin(), padding_dims.begin(), padding_dims.end());
  }
  int64_t broadcast_dim = -1;
  for (int64_t i = SizeToLong(output_shape.size()) - 1; i >= 0; --i) {
    if (pad_input_shape[i] != output_shape[i]) {
      broadcast_dim = i;
      break;
    }
  }
  size_t block_size = 1;
  for (int64_t i = broadcast_dim; i < SizeToLong(pad_input_shape.size()); ++i) {
    block_size *= pad_input_shape[i];
  }

  constexpr size_t kContinugousBlockSize = 128;
  if (block_size > kContinugousBlockSize) {
    ContinugousBroadCastTo(input, output, broadcast_dim, block_size, pad_input_shape, output_shape);
  } else {
    UncontinugousBroadCastTo(input, output, pad_input_shape, output_shape);
  }
}

#define INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, VALUE_DTYPE, S, T) \
  {                                                                \
    KernelAttr()                                                   \
      .AddInputAttr(SELF_DTYPE)                                    \
      .AddInputAttr(VALUE_DTYPE)                                   \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)            \
      .AddOutputAttr(SELF_DTYPE)                                   \
      .AddOutInRef(0, 0),                                          \
      &InplaceCopyCpuKernelMod::LaunchKernel<S, T>                 \
  }

#define INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(SELF_DTYPE, S)                                 \
  INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeBool, S, bool),                       \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeUInt8, S, uint8_t),                 \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeUInt16, S, uint16_t),               \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeUInt32, S, uint32_t),               \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeUInt64, S, uint64_t),               \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeInt8, S, int8_t),                   \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeInt16, S, int16_t),                 \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeInt32, S, int32_t),                 \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeInt64, S, int64_t),                 \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeFloat16, S, float16),               \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeBFloat16, S, bfloat16),             \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeFloat32, S, float),                 \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeFloat64, S, double),                \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeComplex64, S, std::complex<float>), \
    INPLACE_COPY_CPU_KERNEL_REG(SELF_DTYPE, kNumberTypeComplex128, S, std::complex<double>)

using KernelRunFunc = InplaceCopyCpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &InplaceCopyCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeBool, bool),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeUInt8, uint8_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeUInt16, uint16_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeUInt32, uint32_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeUInt64, uint64_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeInt8, int8_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeInt16, int16_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeInt32, int32_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeInt64, int64_t),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeFloat16, float16),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeBFloat16, bfloat16),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeFloat32, float),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeFloat64, double),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeComplex64, std::complex<float>),
    INPLACE_COPY_CPU_KERNEL_REG_BY_SELF(kNumberTypeComplex128, std::complex<double>),
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InplaceCopy, InplaceCopyCpuKernelMod);
}  // namespace inplace_copy_cpu
}  // namespace kernel
}  // namespace mindspore
