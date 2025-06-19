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

#ifndef MINDSPORE_CORE_IR_TENSOR_H_
#define MINDSPORE_CORE_IR_TENSOR_H_

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <algorithm>
#include <iomanip>
#include "ir/device_sync.h"
#include "ir/meta_tensor.h"
#include "utils/log_adapter.h"
#include "base/float16.h"
#include "base/bfloat16.h"
#include "base/float8_e5m2.h"
#include "base/float8_e4m3fn.h"
#include "base/hifloat8.h"
#include "utils/shape_utils.h"
#include "utils/ms_exception.h"
#include "ir/device_event.h"
#include "utils/os.h"
#include "ir/meta_grad_data.h"
#include "ir/tensor_data.h"
#include "utils/ms_utils_secure.h"
#include "base/complex_storage.h"
#include "utils/temp_file_manager.h"
#include "utils/system/env.h"
#include "ir/quantization_param.h"
#include "ir/dtype/op_dtype.h"

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
enum TensorSyncStatus {
  kNoNeedSync,
  kNeedSyncHostToDevice,
  kNeedSyncHostToDeviceImmediately,
  kNeedSyncDeviceToHost,
  kNeedSyncDeviceToHostImmediately
};

enum TensorCompressionType {
  kNoCompression = 0,
  kIndexing = 1,
  kSparse = 2,
  kFSE = 3,
  kBitPacking = 4,
  kFSEInt = 5,
  kFSEInfer = 6
};
// Pinned memory register interface.
class MS_CORE_API PinnedMemRegister {
 public:
  /// \brief Default constructor for register.
  PinnedMemRegister() = default;

  /// \brief Virtual destructor for register.
  virtual ~PinnedMemRegister() = default;

  /// \brief Register pinned memory.
  ///
  /// \param[in] addr The host address to pin.
  /// \param[in] size The host data size.
  /// \return Void.
  virtual void RegisterPinnedMem(void *addr, size_t size) = 0;

  /// \brief UnRegister pinned memory.
  ///
  /// \param[in] addr The host address to unpin.
  /// \return Void.
  virtual void UnRegisterPinnedMem(void *addr) = 0;
};

// A sub namespace in ME to support tensor related definition.
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;

constexpr auto kEllipsis = "...";
constexpr auto kThreshold = 6;
constexpr auto kThreshold1D = 1000;

constexpr auto kThreshold1DFloat = kThreshold * 2;
constexpr auto kThreshold1DInt = kThreshold * 4;
constexpr auto kThreshold1DBool = kThreshold * 2;

template <typename T>
inline constexpr bool IsNonTrivialCastType = false;

template <>
inline constexpr bool IsNonTrivialCastType<float16> = true;
template <>
inline constexpr bool IsNonTrivialCastType<float8_e4m3fn> = true;
template <>
inline constexpr bool IsNonTrivialCastType<float8_e5m2> = true;
template <>
inline constexpr bool IsNonTrivialCastType<hifloat8> = true;
template <>
inline constexpr bool IsNonTrivialCastType<bfloat16> = true;
template <>
inline constexpr bool IsNonTrivialCastType<ComplexStorage<float>> = true;
template <>
inline constexpr bool IsNonTrivialCastType<ComplexStorage<double>> = true;

template <typename T, typename U>
std::unique_ptr<T[]> NewData(const U *input, size_t size) {
  if (input == nullptr || size == 0) {
    return nullptr;
  }
  if (size > INT32_MAX) {
    MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << size * sizeof(T);
  }

  auto data = std::make_unique<T[]>(size);
  if constexpr (!std::is_same_v<T, U> && (IsNonTrivialCastType<T> || IsNonTrivialCastType<U>)) {
    // Because float16 and bfloat16 do not support implicit cast from/to other types,
    // We can not use std::copy() on array of float16 and bfloat16, use a loop here.
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<T>(input[i]);
    }
  } else {
    // otherwise, use std::copy for better performance.
    std::copy(input, input + size, data.get());
  }
  return data;
}

template <typename T, typename Scalar>
std::unique_ptr<T[]> NewData(Scalar scalar) {
  auto data = std::make_unique<T[]>(1);
  data[0] = static_cast<T>(scalar);
  return data;
}

template <typename T>
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, TypeId data_type) {
  const size_t size = SizeOf(shape);
  switch (data_type) {
    case kNumberTypeBool: {
      auto buf = static_cast<bool *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt4: {
      auto buf = static_cast<int8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat32: {
      auto buf = static_cast<float *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat64: {
      auto buf = static_cast<double *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat8E4M3FN: {
      auto buf = static_cast<float8_e4m3fn *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat8E5M2: {
      auto buf = static_cast<float8_e5m2 *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeHiFloat8: {
      auto buf = static_cast<hifloat8 *>(data);
      return NewData<T>(buf, size);
    }
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16: {
      auto buf = static_cast<bfloat16 *>(data);
      return NewData<T>(buf, size);
    }
#endif
    case kNumberTypeComplex64: {
      auto buf = static_cast<ComplexStorage<float> *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeComplex128: {
      auto buf = static_cast<ComplexStorage<double> *>(data);
      return NewData<T>(buf, size);
    }
    case kObjectTypeString: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

template <typename T>
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, size_t data_len) {
  size_t size = SizeOf(shape);
  if (size * sizeof(T) != data_len) {
    MS_LOG(EXCEPTION) << "Incorrect tensor input data length " << data_len << ", expect " << size * sizeof(T)
                      << " item size " << sizeof(T);
  }
  auto buf = static_cast<T *>(data);
  return NewData<T>(buf, size);
}

// TensorStringifier provide methods to convert tensor data to its string representation.
template <typename T>
class TensorStringifier {
 public:
  TensorStringifier(const T *data, size_t data_size, size_t ndim) : data_(data), data_size_(data_size), ndim_(ndim) {}
  ~TensorStringifier() = default;

  std::string ToString(TypeId, const ShapeVector &shape, bool use_comma) const {
    constexpr auto valid =
      std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value ||
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value ||
      std::is_same<T, float8_e4m3fn>::value || std::is_same<T, float8_e5m2>::value ||
      std::is_same<T, hifloat8>::value || std::is_same<T, ComplexStorage<float>>::value ||
      std::is_same<T, ComplexStorage<double>>::value || std::is_same<T, bfloat16>::value;
    static_assert(valid, "Type is invalid");
    if (data_size_ == 0) {
      return "";
    }
    if (data_ == nullptr) {
      return "<uninitialized>";
    }

    std::ostringstream ss;
    if (data_size_ == 1 && ndim_ == 0) {  // Scalar
      int max = 0;
      OutputDataString(ss, 0, 0, 1, false, &max);
      return ss.str();
    }

    int num_width = 0;
    ssize_t cursor = 0;
    SummaryStringRecursive(ss, shape, &cursor, 0, use_comma, &num_width);
    return ProcessPlaceholder(ss, num_width);
  }

 private:
  static void OutputFloatDataString(std::ostringstream &ss, bool isScalar, const T &value) {
    if (isScalar) {
      ss << value;
    } else {
      // The placeholder of float16 is fixed at 11, while float/double is fixed at 15.
      const int width = std::is_same<T, float16>::value ? 11 : 15;
      // The printing precision of float16 is fixed at 4, while float/double is fixed at 8.
      const int precision = std::is_same<T, float16>::value ? 4 : 8;
      ss << std::setw(width) << std::setprecision(precision) << std::setiosflags(std::ios::scientific | std::ios::right)
         << value;
    }
  }

  static void OutputBoolDataString(std::ostringstream &ss, bool isScalar, const T &value) {
    if (isScalar) {
      ss << (value ? "True" : "False");
    } else {
      constexpr int bool_max_width = sizeof("False") - 1;
      ss << std::setw(bool_max_width) << std::setiosflags(std::ios::right) << (value ? "True" : "False");
    }
  }

  static void OutputOtherDataString(std::ostringstream &ss, bool isScalar, const T &value, int *max_width) {
    std::ostringstream value_ss;
    if constexpr (std::is_same<T, uint8_t>::value) {
      value_ss << static_cast<uint16_t>(value);
    } else if constexpr (std::is_same<T, int8_t>::value) {
      value_ss << static_cast<int16_t>(value);
    } else {
      value_ss << value;
    }
    auto value_str = value_ss.str();
    if (!isScalar) {
      const int width = static_cast<int>(value_str.size());
      *max_width = std::max(*max_width, width);
      // Add a padding string before the number, such as "###123", for subsequent replacement.
      std::string pad(width, '#');
      ss << pad;
    }
    ss << value_str;
  }

  static std::string ProcessPlaceholder(const std::ostringstream &ss, int max_width) {
    std::string str = ss.str();
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, float16>::value || std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
      return str;
    }
    // Replace # with placeholder.
    size_t index = str.find('#');
    while (index != std::string::npos) {
      size_t pos = index;
      while (str[pos] == '#') {
        pos++;
      }
      size_t len = pos - index;
      std::string space(max_width - SizeToInt(len), ' ');
      str = str.replace(index, len, space);
      index = str.find('#', index);
    }
    return str;
  }

  void OutputDataString(std::ostringstream &ss, ssize_t cursor, ssize_t start, ssize_t end, bool use_comma,
                        int *max_width) const {
    const bool isScalar = ndim_ == 0 && end - start == 1;
    constexpr auto isBool = std::is_same<T, bool>::value;
    constexpr auto isFloat =
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value;
    constexpr auto isComplex =
      std::is_same<T, ComplexStorage<float>>::value || std::is_same<T, ComplexStorage<double>>::value;
    constexpr int linefeedThreshold = isFloat ? kThreshold1DFloat : (isBool ? kThreshold1DBool : kThreshold1DInt);
    for (ssize_t i = start; i < end && (cursor + i) < static_cast<ssize_t>(data_size_); i++) {
      const auto value = data_[cursor + i];
      if constexpr (isComplex) {
        ss << value;
      } else if constexpr (isFloat) {
        OutputFloatDataString(ss, isScalar, value);
      } else if (isBool) {
        OutputBoolDataString(ss, isScalar, value);
      } else {
        OutputOtherDataString(ss, isScalar, value, max_width);
      }
      if (!isScalar && i != end - 1) {
        if (use_comma) {
          ss << ',';
        }
        ss << ' ';
      }
      if (!isScalar && ndim_ == 1 && end - start > (kThreshold >> 1) && (i + 1) % linefeedThreshold == 0) {
        // Add a line feed every {threshold of type} for 1D tensor.
        ss << '\n' << ' ';
      }
    }
  }

  void SummaryStringRecursive(std::ostringstream &ss, const ShapeVector &shape, ssize_t *cursor, ssize_t depth,
                              bool use_comma, int *max_width) const {
    if (depth >= static_cast<ssize_t>(ndim_)) {
      return;
    }
    ss << '[';
    if (depth == static_cast<ssize_t>(ndim_) - 1) {  // Bottom dimension
      ssize_t num = shape[depth];
      if ((num > kThreshold && ndim_ > 1) || (num > kThreshold1D && ndim_ == 1)) {
        OutputDataString(ss, *cursor, 0, kThreshold >> 1, use_comma, max_width);
        ss << ' ' << kEllipsis << ' ';
        OutputDataString(ss, *cursor, num - (kThreshold >> 1), num, use_comma, max_width);
      } else {
        OutputDataString(ss, *cursor, 0, num, use_comma, max_width);
      }
      *cursor += num;
    } else {  // Middle dimension
      ssize_t num = shape[depth];
      // Handle the first half.
      for (ssize_t i = 0; i < std::min(static_cast<ssize_t>(kThreshold >> 1), num); i++) {
        if (i > 0) {
          if (use_comma) {
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
        }
        SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
      }
      // Handle the ignored part.
      if (num > kThreshold) {
        if (use_comma) {
          ss << ',';
        }
        ss << '\n';
        ss << std::setw(depth + 1) << ' ';  // Add the indent.
        ss << kEllipsis;
        // Ignored at this layer.
        ssize_t ignored = shape[depth + 1];
        const size_t offset = 2;
        for (ssize_t i = depth + offset; i < static_cast<ssize_t>(ndim_); i++) {
          ignored *= shape[i];
        }
        // Multiple with ignored layers number.
        ignored *= (num - kThreshold);
        *cursor += ignored;
      }
      // Handle the second half.
      if (num > (kThreshold >> 1)) {
        ssize_t iter_times =
          std::min(static_cast<ssize_t>(num - (kThreshold >> 1)), static_cast<ssize_t>(kThreshold >> 1));
        for (ssize_t i = 0; i < iter_times; i++) {
          if (use_comma && (i != 0 || num <= kThreshold)) {  // Not just after ignored part || Not handle ignored part
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
          SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
        }
      }
    }
    ss << ']';
  }

  const T *data_;
  const size_t data_size_;
  const size_t ndim_;
};
// Tensor data implementation.
template <typename T>
class TensorDataImpl : public TensorData {
 public:
  explicit TensorDataImpl(const ShapeVector &shape) : ndim_(shape.size()), data_size_(SizeOf(shape)) {}
  ~TensorDataImpl() override {
    try {
      RemoveOffloadFile();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception occurred when cleaning tensor. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Exception occurred when cleaning tensor.";
    }
  }

  TensorDataImpl(const ShapeVector &shape, void *data, size_t data_len)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_len)) {}

  TensorDataImpl(const ShapeVector &shape, void *data, TypeId data_type)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_type)) {}

  template <typename U>
  TensorDataImpl(const ShapeVector &shape, const U *input, size_t size)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(input, size)) {}

  template <typename Scalar>
  TensorDataImpl(const ShapeVector &shape, Scalar scalar)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(scalar)) {}

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  bool is_sub_data() const override { return false; }

  bool has_sub_data() const override { return false; }

  void *data() override {
    if (data_ != nullptr) {
      return data_.get();
    }

    if (data_size_ > INT32_MAX) {
      MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << data_size_ * sizeof(T);
    }
    // Lazy allocation.
    data_ = std::make_unique<T[]>(data_size_);

    // Load data from file
    if (!file_path_.empty()) {
      auto fs = mindspore::system::Env::GetFileSystem();
      MS_EXCEPTION_IF_NULL(fs);
      if (fs->FileExist(file_path_)) {
        auto file = fs->CreateWriteFile(file_path_, "r+");
        MS_EXCEPTION_IF_NULL(file);
        bool success = file->PRead(data_.get(), data_size_ * sizeof(T), 0);
        if (!success) {
          MS_LOG(WARNING) << "Tensor load data from file: " << file_path_ << " failed!";
        }
        if (!file->Close()) {
          MS_LOG(WARNING) << "Close tensor file: " << file_path_ << " failed!";
        }
      } else {
        MS_LOG(WARNING) << "Invalid tensor file path: " << file_path_;
      }
    }
    return data_.get();
  }

  void set_file_path(const std::string &file_path) override { file_path_ = file_path; }

  const std::string file_path() const override { return file_path_; }

  const void *const_data() const override {
    // May return nullptr if data not initialized.
    return data_.get();
  }

  virtual bool equals(const TensorDataImpl<T> &other) const {
    auto ptr = &other;
    if (ptr == this) {
      return true;
    }
    if (data_ == nullptr || ptr->data_ == nullptr) {
      return false;
    }
    return (ndim_ == ptr->ndim_) && (data_size_ == ptr->data_size_) &&
           std::equal(data_.get(), data_.get() + data_size_, ptr->data_.get());
  }

  bool equals(const TensorData &other) const override {
    // Not same type, compare data byte by byte.
    return TensorData::equals(other);
  }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override {
    TensorStringifier<T> stringifier{data_.get(), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }

 private:
  void RemoveOffloadFile() {
    if (!file_path_.empty()) {
      TempFileManager::GetInstance().RemoveFile(file_path_);
      TempFileManager::GetInstance().UnRegister(file_path_);
      file_path_ = "";
    }
  }

  size_t ndim_{0};
  size_t data_size_{0};
  std::unique_ptr<T[]> data_;
  std::string file_path_{""};
};
template <template <class> class ImplClass = TensorDataImpl, typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, Args &&... args) {
  switch (data_type) {
    case kNumberTypeBool:
      return std::make_shared<ImplClass<bool>>(std::forward<Args>(args)...);
    case kNumberTypeUInt8:
      return std::make_shared<ImplClass<uint8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt4:
      return std::make_shared<ImplClass<int8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt8:
      return std::make_shared<ImplClass<int8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt16:
      return std::make_shared<ImplClass<int16_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt:
    case kNumberTypeInt32:
      return std::make_shared<ImplClass<int32_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt64:
      return std::make_shared<ImplClass<int64_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt16:
      return std::make_shared<ImplClass<uint16_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt32:
      return std::make_shared<ImplClass<uint32_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt64:
      return std::make_shared<ImplClass<uint64_t>>(std::forward<Args>(args)...);
    case kNumberTypeFloat16:
      return std::make_shared<ImplClass<float16>>(std::forward<Args>(args)...);
    case kNumberTypeFloat:
      return std::make_shared<ImplClass<float>>(std::forward<Args>(args)...);
    case kNumberTypeFloat32:
      return std::make_shared<ImplClass<float>>(std::forward<Args>(args)...);
    case kNumberTypeFloat64:
      return std::make_shared<ImplClass<double>>(std::forward<Args>(args)...);
    case kNumberTypeFloat8E4M3FN:
      return std::make_shared<ImplClass<float8_e4m3fn>>(std::forward<Args>(args)...);
    case kNumberTypeFloat8E5M2:
      return std::make_shared<ImplClass<float8_e5m2>>(std::forward<Args>(args)...);
    case kNumberTypeHiFloat8:
      return std::make_shared<ImplClass<hifloat8>>(std::forward<Args>(args)...);
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16:
      return std::make_shared<ImplClass<bfloat16>>(std::forward<Args>(args)...);
#endif
    case kNumberTypeComplex64:
      return std::make_shared<ImplClass<ComplexStorage<float>>>(std::forward<Args>(args)...);
    case kNumberTypeComplex128:
      return std::make_shared<ImplClass<ComplexStorage<double>>>(std::forward<Args>(args)...);
    case kObjectTypeString:
      return std::make_shared<ImplClass<uint8_t>>(std::forward<Args>(args)...);
    case kObjectTypeTensorType:
    case kObjectTypeMapTensorType:
      return std::make_shared<ImplClass<int>>(std::forward<Args>(args)...);
    default:
      break;
  }
  MS_LOG(ERROR) << "Cannot construct Tensor because of unsupported data type: " << TypeIdToString(data_type) << ".";
  return nullptr;
}

struct Version {
 public:
  Version() : version_counter_(std::make_shared<uint32_t>(0)) {}
  void BumpVersion() { (*version_counter_)++; }
  uint32_t current_version() const {
    MS_EXCEPTION_IF_NULL(version_counter_);
    return *version_counter_;
  }

 private:
  std::shared_ptr<uint32_t> version_counter_;
};

template <typename T>
class FutureData {
 public:
  FutureData(std::shared_ptr<T> data, std::exception_ptr e_ptr) : data_(std::move(data)), e_ptr_(std::move(e_ptr)) {}
  virtual ~FutureData() {}

  virtual std::shared_ptr<T> GetData() const { return data_; }
  const std::exception_ptr &GetException() const { return e_ptr_; }

 private:
  std::shared_ptr<T> data_;
  std::exception_ptr e_ptr_;
};

template <typename T>
class FutureBase {
 public:
  explicit FutureBase(std::future<std::shared_ptr<tensor::FutureData<T>>> future) : future_(std::move(future)) {}
  virtual ~FutureBase() {}
  virtual std::shared_ptr<T> Get() = 0;

 protected:
  std::future<std::shared_ptr<tensor::FutureData<T>>> future_;
  std::shared_ptr<tensor::FutureData<T>> future_data_;
};
// brief Device info of Tensor
//
// Includes the format, data type and host format of a tensor.
struct DeviceInfo {
  explicit DeviceInfo(const std::unique_ptr<DeviceInfo> &device_info) {
    if (device_info == nullptr) {
      return;
    }
    format_ = device_info->format_;
    data_type_ = device_info->data_type_;
    host_format_ = device_info->host_format_;
    device_id_ = device_info->device_id_;
  }
  explicit DeviceInfo(std::string format = "DefaultFormat", TypePtr data_type = nullptr,
                      std::string host_format = "DefaultFormat", int32_t device_id = 0)
      : format_(std::move(format)),
        data_type_(std::move(data_type)),
        host_format_(std::move(host_format)),
        device_id_(device_id) {}
  std::string format_ = "DefaultFormat";
  TypePtr data_type_ = nullptr;
  std::string host_format_ = "DefaultFormat";
  int32_t device_id_ = 0;
};

// Tensor entity class
class MS_CORE_API Tensor : public MetaTensor {
 public:
  Tensor() = default;

  /// \brief Create tensor from another tensor, data is shared.
  ///
  /// \param[in] tensor [Tensor] The input tensor.
  explicit Tensor(const Tensor &tensor);
  /// \brief Create tensor with given data type from another tensor.
  ///
  /// \param[in] tensor [Tensor] The input tensor.
  /// \param[in] data_type [TypeId] The new tensor data type.
  Tensor(const Tensor &tensor, TypeId data_type);

  /// \brief Create tensor with the given shared tensor data.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The shared tensor data.
  Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data);

  /// \brief Create a lazy allocated tensor.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  Tensor(TypeId data_type, const ShapeVector &shape);

  /// \brief Create a tensor with input data buffer.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The input data to be copied into tensor.
  /// \param[in] data_len The length of data in bytes.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len);

  /// \brief Create a tensor with input data buffer and given source data type.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The input data to be copied into tensor.
  /// \param[in] src_data_type The source data type.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type);

  /// \brief Create 1 dimension tensor from an int vector.
  ///
  /// \param[in] input [std::vector<int64_t>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensor from an int vector.
  ///
  /// \param[in] input [std::vector<int32_t>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<int32_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensor from a float vector.
  ///
  /// \param[in] input [std::vector<double>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<double> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensor from a float vector.
  ///
  /// \param[in] input [std::vector<float>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<float> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int64_t scalar.
  ///
  /// \param[in] input [int64] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int64_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int32_t scalar.
  ///
  /// \param[in] input [int32] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int32_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int16_t scalar.
  ///
  /// \param[in] input [int16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int16_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int8_t scalar.
  ///
  /// \param[in] input [int8] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int8_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a double scalar.
  ///
  /// \param[in] input [double] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(double input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float scalar.
  ///
  /// \param[in] input [float] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float16 scalar.
  ///
  /// \param[in] input [float16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float16 input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float8_e5m2 scalar.
  ///
  /// \param[in] input [float8_e5m2] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float8_e5m2 input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float8_e4m3fn scalar.
  ///
  /// \param[in] input [float8_e4m3fn] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float8_e4m3fn input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a hifloat8 scalar.
  ///
  /// \param[in] input [hifloat8] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(hifloat8 input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a bfloat16 scalar.
  ///
  /// \param[in] input [bfloat16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(bfloat16 input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint64 scalar.
  ///
  /// \param[in] input [uint64] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint64_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint32 scalar.
  ///
  /// \param[in] input [uint32] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint32_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint16 scalar.
  ///
  /// \param[in] input [uint16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint16_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint8 scalar.
  ///
  /// \param[in] input [uint8] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint8_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a bool scalar.
  ///
  /// \param[in] input [bool] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(bool input, const TypePtr &data_type = nullptr);

  /// \brief Create a chunk tensor with the given data size.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] data_size The tensor chunk data size in number of elements.
  Tensor(TypeId data_type, size_t data_size);

  /// \brief Create a Tensor which shape and size may be inconsistent, such as Tensor with compression data.
  ///
  /// \param[in] origin_data_type [TypeId] Data type of the origin tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] compression_data_size The compression data buffer size.
  /// \param[in] TensorCompressionType The tensor compression type.
  Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
         TensorCompressionType compression_type);

  Tensor &operator=(const Tensor &tensor);

  /// Destructor of Tensor.
  ~Tensor() override;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  /// \brief Compare two tensor objects to see if they have same data type, shape and data address.
  ///
  /// \param[in] tensor The Tensor object to be compared.
  /// \return True if having same type, shape and data address, otherwise false.
  bool operator==(const Tensor &tensor) const;

  /// \brief Create Abstract for Tensor.
  ///
  /// \return Abstract of Tensor.
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief It is different from 'operator==' which just compares shape/type/address,
  /// it does real value comparison.
  ///
  /// \param[in] tensor The Tensor object to be compared.
  /// \return True if it has the same value, otherwise false.
  bool ValueEqual(const Tensor &tensor) const;

  /// \brief Assign value to this tensor.
  ///
  /// \param[in] tensor The input tensor.
  /// \return Tensor with new value.
  Tensor &AssignValue(const Tensor &tensor);

  bool operator==(const Value &other) const override {
    if (other.isa<Tensor>()) {
      auto &other_ = static_cast<const Tensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Gets tensor's dimension.
  ///
  /// \return The number of dimensions of the tensor data.
  int DataDim() const { return static_cast<int>(data().ndim()); }

  /// \brief Getting tensor data size.
  ///
  /// \return The total number of elements of the tensor data.
  size_t DataSize() const { return data().size(); }

  /// \brief Get the data type of the tensor for C++
  ///
  /// \return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const { return static_cast<int>(data_type_); }

  /// \brief Get the tensor's shape for C++
  ///
  /// \return [ShapeVector]
  ShapeVector shape_c(void) const { return shape(); }

  /// \brief Get Tensor data pointer for c++ type
  ///
  /// \return The pointer to the object
  void *data_c() { return data().data(); }

  /// \brief Get Tensor data byte-size for c++ type
  ///
  /// \return byte size of Tensor data
  size_t Size() const { return static_cast<size_t>(data().nbytes()); }

  /// \brief The pointer to the object
  void *data_c() const { return data_->data(); }

  /// \brief To synchronize data with the device, you need to wait for the data to be valid.
  ///
  void data_sync(bool need_wait = true, bool inpalce = true, bool sync_on_demand = false) const;

  /// \brief Get the internal data object.
  ///
  /// \return The reference to internal data object.
  TensorData &data() {
    MS_EXCEPTION_IF_NULL(data_);
    return *data_;
  }

  /// \brief Get the internal data shared pointer.
  ///
  /// return The reference to internal data object.
  const TensorDataPtr &data_ptr() const { return data_; }

  /// \brief Get the internal data object.
  ///
  /// \return The reference to internal data object.
  const TensorData &data() const { return *data_; }

  void set_data(const TensorDataPtr &data) { data_ = data; }

  TypeId set_data_type(TypeId data_type) override;

  size_t set_shape(const ShapeVector &shape) override;

  /// \brief Get information about shape and data type.
  ///
  /// \return Information about shape and data type.
  std::string GetShapeAndDataTypeInfo() const;

  /// \brief Get display information of limit size.
  ///
  /// \param[in] limit_size The limit size.
  /// \return The display information of limit size.
  std::string ToStringInternal(size_t limit_size) const;

  /// \brief Get display information with unlimited size.
  ///
  /// \return The display information with unlimited size.
  std::string ToStringNoLimit() const;

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  /// \brief Get display information in repr form.
  ///
  /// \return The display information in repr form.
  std::string ToStringRepr() const;

  /// \brief Check if this Tensor is forward output.
  ///
  /// \return Whether this Tensor is forward output.
  bool is_forward_output() const { return is_forward_output_; }

  /// \brief Set the forward output flag of this Tensor.
  ///
  /// \param[in] is_forward_output Whether this Tensor is forward output.
  void set_is_forward_output(bool is_forward_output) { is_forward_output_ = is_forward_output; }

  /// \brief Check if this Tensor is used in bprop graph.
  ///
  /// \return Whether this Tensor is used in bprop graph.
  bool used_in_bprop_graph() const { return used_in_bprop_graph_; }

  /// \brief Set used in bprop graph flag of this Tensor.
  ///
  /// \param[in] used_in_bprop_graph Whether this Tensor is forward output.
  void set_used_in_bprop_graph(bool used_in_bprop_graph) { used_in_bprop_graph_ = used_in_bprop_graph; }

  /// \brief Get the device address.
  ///
  /// \return The device address.
  DeviceSyncPtr device_address() const;

  /// \brief Set the device address.
  ///
  /// \param[in] device_sync The input Device synchronization.
  /// \param[in] need_update_ref_count If need_update_ref_count is true, the device address cannot be released and
  /// reused, so the feature map should set false when set device address of tensor.
  void set_device_address(const DeviceSyncPtr &device_sync, bool need_update_ref_count = true);

  /// \brief Get the id of this Tensor.
  ///
  /// \return The id of this Tensor.
  uint64_t id() const { return id_; }

  /// \brief Set lazy callback function to this Tensor
  ///
  /// \param[in] lazy_callback Wait for async tasks finish before data_sync.
  static void RegisterLazyCallback(const std::function<void(void)> &lazy_callback) { lazy_callback_ = lazy_callback; }

  /// \brief Contiguous callback function to this Tensor
  ///
  /// \return The contiguous callback function
  const std::function<DeviceSyncPtr(const DeviceSyncPtr &)> &contiguous_callback() { return contiguous_callback_; }

  /// \brief Set contiguous callback function to this Tensor
  ///
  /// \param[in] contiguous_callback The callback from backend when need to make tensor contiguous.
  void set_contiguous_callback(const std::function<DeviceSyncPtr(const DeviceSyncPtr &)> &contiguous_callback) {
    contiguous_callback_ = contiguous_callback;
  }

  /// @brief Get Pynative auto_grad meta data.
  /// @return Auto grad meta data
  const AutoGradMetaInterfacePtr &auto_grad_meta_data() const { return auto_grad_meta_data_; }

  /// @brief Set Pynative auto_grad meta data.
  /// @param auto_grad_meta_data
  void set_auto_grad_meta_data(const AutoGradMetaInterfacePtr &auto_grad_meta_data) {
    auto_grad_meta_data_ = auto_grad_meta_data;
  }

  /// \brief Check whether the tensor is used in auto grad.
  ///
  /// \return Boolean indicate whether the tensor is used in auto grad.
  bool HasAutoGrad() const {
    return auto_grad_meta_data_ != nullptr && auto_grad_meta_data_->input_type() == InputType::kOpOutput;
  }

  /// \brief Get tensor storage info.
  ///
  /// \return Tensor storage info, the value is nullptr default.
  const TensorStorageInfoPtr storage_info() const;

  /// \brief Set tensor storage info.
  ///
  void set_storage_info(const TensorStorageInfoPtr &storage_info);

  /// \brief Set synchronization status.
  ///
  /// \param[in] sync_status The input synchronization status.
  void set_sync_status(TensorSyncStatus sync_status) const { sync_status_ = sync_status; }

  /// \brief Get synchronization status.
  ///
  /// \return The synchronization status.
  TensorSyncStatus sync_status() const { return sync_status_; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncDeviceToHostImmediately.
  bool NeedSyncDeviceToHostImmediately() const { return sync_status_ == kNeedSyncDeviceToHostImmediately; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncDeviceToHost.
  bool NeedSyncDeviceToHost() const { return sync_status_ == kNeedSyncDeviceToHost; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncHostToDevice.
  bool NeedSyncHostToDevice() const { return sync_status_ == kNeedSyncHostToDevice; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncHostToDeviceImmediately.
  bool NeedSyncHostToDeviceImmediately() const { return sync_status_ == kNeedSyncHostToDeviceImmediately; }

  /// \brief Get tensor's BaseShape.
  ///
  /// \return The BaseShape of this tensor.
  const BaseShapePtr &base_shape_ptr() const { return base_shape_ptr_; }

  /// \brief Set tensor's BaseShape.
  ///
  /// \param[in] BaseShapePtr The tensor's BaseShape.
  void set_base_shape(const BaseShapePtr &base_shape) { base_shape_ptr_ = base_shape; }

  /// \brief Determines whether the memory of tensor is contiguous.
  ///
  /// \return True if tensor memory is contiguous, false otherwise.
  bool is_contiguous() const;

  bool NeedContiguous() const;

  /// \brief Get tensor storage stride.
  ///
  /// \return storage stride.
  std::vector<int64_t> stride() const;

  /// \brief Get tensor storage offset.
  ///
  /// \return storage offset.
  const int64_t storage_offset() const;

  /// \brief Get version.
  ///
  /// \return storage offset.
  const Version &version() const { return version_; }

  /// \brief Set version.
  ///
  void set_version(const Version &version) { version_ = version; }

  /// \brief Bump version.
  ///
  void BumpVersion() { version_.BumpVersion(); }

  void set_need_pipeline_sync(bool need_pipeline_sync) { need_pipeline_sync_ = need_pipeline_sync; }

  bool need_pipeline_sync() const { return need_pipeline_sync_; }

  /// \brief Execute lazy task.
  ///
  void ExecuteLazyTask() const;

  /// \brief Execute contiguous callback.
  ///
  DeviceSyncPtr CallContiguousCallback() const;

  /// \brief To synchronize data with the device without keeping device address, you need to wait for the data to be
  /// valid.
  ///
  void data_sync_directly(const DeviceSync *const device_sync, bool need_wait = true) const;

  /// \brief Check if this Tensor is initialized.
  ///
  /// \return Whether this Tensor is initialized.
  bool is_init() const { return init_flag_; }

  /// \brief Set the initialization flag of this Tensor.
  ///
  /// \param[in] flag Whether this Tensor is initialized.
  void set_init_flag(bool flag) { init_flag_ = flag; }

  /// \brief Get the cast dtype of this Tensor.
  ///
  /// \return The cast dtype of this Tensor.
  TypePtr cast_dtype() { return cast_dtype_; }

  /// \brief Set the cast dtype of this Tensor.
  ///
  /// \param[in] dtype The input cast dtype.
  void set_cast_dtype(const TypePtr &dtype = nullptr) { cast_dtype_ = dtype; }

  /// \brief Used cache_enable to update the tensor from the cache to the host.
  ///
  /// \return True if caching is enabled, otherwise false.
  bool cache_enable() const { return cache_enable_; }

  /// \brief Set cache_enable.
  ///
  /// \param[in] cache_enable Whether to enable caching.
  void set_cache_enable(bool cache_enable = true) { cache_enable_ = cache_enable; }

  /// \brief Get the pointer of hashmap tensor.
  ///
  /// \return The pointer of hashmap tensor.
  std::shared_ptr<Tensor> hashmap_tensor_ptr() const { return hashmap_tensor_ptr_; }

  /// \brief Set the pointer of hashmap tensor.
  ///
  /// \param[in] hashmap_tensor_ptr The input pointer of hashmap tensor.
  void set_hashmap_tensor_ptr(const std::shared_ptr<Tensor> &hashmap_tensor_ptr = nullptr) {
    hashmap_tensor_ptr_ = hashmap_tensor_ptr;
  }

  /// \brief Get the pointer of cache tensor.
  ///
  /// \return The pointer of cache tensor.
  std::shared_ptr<Tensor> cache_tensor_ptr() const { return cache_tensor_ptr_; }

  /// \brief Set the pointer of cache tensor.
  ///
  /// \param[in] cache_tensor_ptr The input pointer of cache tensor.
  void set_cache_tensor_ptr(const std::shared_ptr<Tensor> &cache_tensor_ptr = nullptr) {
    cache_tensor_ptr_ = cache_tensor_ptr;
  }

  /// \brief Check if this Tensor is the output of graph.
  ///
  /// \return Whether this Tensor is the output of graph
  bool IsGraphOutput() const { return graph_output_; }

  /// \brief Set whether this Tensor is the output of graph.
  void SetIsGraphOutput() { graph_output_ = true; }

  /// \brief Get whether this Tensor is updated by the device.
  ///
  /// \return Whether this Tensor is updated by the device.
  bool IsUpdatedByDevice() const { return updated_by_device_; }

  /// \brief Set whether this Tensor is updated by the device.
  void SetIsUpdateByDevice() { updated_by_device_ = true; }

  /// \brief Get callback need to execute when value is updated of Tensor.
  ///
  /// \return The callback need to execute when value is updated of Tensor.
  const std::function<void(const Tensor *)> &update_value_callback() const { return update_value_callback_; }

  /// \brief Set callback need to execute when value is updated of Tensor.
  ///
  /// \param[in] update_value_callback The callback need to execute when value is updated of Tensor.
  void set_update_value_callback(const std::function<void(const Tensor *)> &update_value_callback) {
    update_value_callback_ = update_value_callback;
  }

  /// \brief Get the memory chunk pointer and offset if memory chunk for this tensor exists.
  ///
  /// \return The memory chunk pointer and offset, nullptr and 0 if no memory chunk exists.
  std::pair<void *, size_t> GetChunkOffset() const;

  /// \brief Reset tensors data so that they are using contiguous memory chunks grouped by data type.
  ///
  /// \param[in] tensors The tensors to be processed.
  /// \param[in] fusion_size Maximum memory chunk size in bytes, 0 for unlimited.
  ///
  /// \return Tensors that data are pointed to each contiguous memory chunks.
  static TensorPtrList FlattenTensors(const TensorPtrList &tensors, size_t fusion_size = 0);

  /// \brief Check if FlattenTensors called for the input tensors.
  ///
  /// \param[in] tensors The tensors to be checked.
  ///
  /// \return True if FlattenTensors called for input tensors, false otherwise.
  static bool IsFlattened(const TensorPtrList &tensors);

  /// \brief Get tensors for each contiguous memory chunks used by the input tensors.
  ///
  /// \param[in] tensors The input tensors.
  ///
  /// \return Tensors that data are pointed to each contiguous memory chunks, empty if failed.
  static TensorPtrList GetFlattenedTensors(const TensorPtrList &tensors);

  /// \brief Get tensor for each contiguous memory chunks used.
  ///
  /// \param[in] tensor The given tensor.
  ///
  /// \return Tensor that data are pointed to each contiguous memory chunks, empty if failed.
  static const TensorPtr GetFlattenedTensor(const TensorPtr &tensor);

  /// \brief Get tensors stub flag.
  ///
  /// \param[in] none.
  ///
  /// \return If compile with backend, return false, else return true.
  static bool CheckStub();

  /// \brief Get the fusion size for the given flat tensors.
  ///
  /// \param[in] flat_tensors The input flat tensors.
  ///
  /// \return fusion size for the given flat tensors.
  static size_t GetFusionSize(const TensorPtrList &flat_tensors);

  /// \brief Get the tensor compression type.
  ///
  /// \return tensor compression type.
  TensorCompressionType compression_type() const { return compression_type_; }

  /// \brief If tensor use persistent tensor data.
  ///
  /// \return if use persistent tenor data.
  bool is_persistent_data() const;

  /// \brief Set tensor name.
  ///
  /// \param[in] tensor_name The tensor name.
  void set_name(const std::string &tensor_name) { tensor_name_ = tensor_name; }

  /// \brief Get the tensor name.
  ///
  /// \return tensor name.
  const std::string &name() const { return tensor_name_; }

  /// \brief Set tensor quant param.
  ///
  /// \param[in] quant_param The tensor quant param.
  void set_quant_param(const std::vector<std::shared_ptr<QuantizationParam>> &quant_params) {
    quant_params_.assign(quant_params.begin(), quant_params.end());
  }

  /// \brief Get the tensor quant param.
  ///
  /// \return tensor quant param.
  const std::vector<std::shared_ptr<QuantizationParam>> &quant_params() const { return quant_params_; }

  /// \brief Offload tensor to file.
  ///
  /// \return offload tensor success.
  bool Offload(const std::string &file_path);

  /// \brief Get tensor offload file path.
  ///
  /// \return offload file path, or empty string if tensor has not offload.
  const std::string GetOffloadFilePath() const;

  /// \brief pin tensor memory.
  ///
  /// \param[in] register to pin tensor data.
  void PinMemory(PinnedMemRegister *pin_mem_register);

  /// \brief unpin tensor memory.
  void UnPinMemory();

  /// \brief Get tensor's device info.
  ///
  /// \return The device info of this tensor.
  DeviceInfo device_info() const {
    if (device_info_ != nullptr) {
      return *device_info_;
    }
    device_info_ = std::make_unique<DeviceInfo>();
    return *device_info_;
  }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] device_info The tensor's device info.
  void set_device_info(const DeviceInfo &device_info) { device_info_ = std::make_unique<DeviceInfo>(device_info); }

  ops::OP_DTYPE source_type() const { return source_type_; }

  void set_source_type(ops::OP_DTYPE source_type) { source_type_ = source_type; }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] format The input format.
  /// \param[in] data_type The input data type.
  /// \param[in] host_format The input host format.
  void SetDeviceInfo(const std::string &format, const TypePtr &data_type,
                     const std::string &host_format = "DefaultFormat");

  void set_copy_done_flag(bool flag) { copy_done_flag_ = flag; }
  bool get_copy_done_flag() const { return copy_done_flag_; }

 private:
  // Really execute callback function when host value is updated of Tensor.
  void ExecuteUpdateValueCallback() const;

  // function size 32
  inline static std::function<void(void)> lazy_callback_{nullptr};
  std::function<DeviceSyncPtr(const DeviceSyncPtr &)> contiguous_callback_{nullptr};
  std::function<void(const Tensor *)> update_value_callback_{nullptr};

  // string size 32
  uint64_t id_;
  std::string tensor_name_;

  // shared_ptr size 16
  Version version_{};
  mutable DeviceSyncPtr device_sync_{nullptr};
  AutoGradMetaInterfacePtr auto_grad_meta_data_{nullptr};
  TensorStorageInfoPtr storage_info_;
  TensorDataPtr data_{nullptr};
  // Tensor base shape which contain dynamic shape info.
  BaseShapePtr base_shape_ptr_{nullptr};
  std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  TypePtr cast_dtype_{nullptr};
  std::vector<std::shared_ptr<QuantizationParam>> quant_params_;

  // pointer size 8
  ops::OP_DTYPE source_type_{ops::OP_DTYPE::DT_BEGIN};
  mutable std::unique_ptr<DeviceInfo> device_info_;
  PinnedMemRegister *pin_mem_register_{nullptr};

  // enum size 4
  mutable TensorSyncStatus sync_status_{kNeedSyncHostToDevice};
  TensorCompressionType compression_type_{kNoCompression};

  // bool size 1
  bool is_forward_output_{false};
  bool used_in_bprop_graph_{true};
  bool need_pipeline_sync_{false};
  bool init_flag_{false};
  bool graph_output_{false};
  bool updated_by_device_{false};
  bool cache_enable_{false};
  bool copy_done_flag_{false};
};

// CSRTensor entity class
class MS_CORE_API CSRTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create CSRTensor with given data type from another tensor.
  ///
  /// \param[in] indptr [Tensor] The indices pointer.
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the CSRensor.
  CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape);

  /// Destructor of CSRTensor.
  ~CSRTensor() override = default;

  MS_DECLARE_PARENT(CSRTensor, MetaSparseTensor)

  /// \brief Gets CSRTensor's indptr.
  ///
  /// \return [TensorPtr] The indices pointer.
  TensorPtr GetIndptr() { return indptr_; }

  /// \brief Gets CSRTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets CSRTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  /// \brief Compare two csrtensor objects to see if they have same data address.
  ///
  /// \param[in] csr_tensor The csrtensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const CSRTensor &csr_tensor) const { return &csr_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<CSRTensor>()) {
      auto &other_ = static_cast<const CSRTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  const size_t GetSizeAt(size_t index) const;

  TensorPtr GetTensorAt(size_t index) const;

  const size_t GetTensorLength() const { return kShapeIdx + shape().size(); }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  static constexpr size_t kIndptrIdx = 0;
  static constexpr size_t kIndicesIdx = 1;
  static constexpr size_t kValuesIdx = 2;
  static constexpr size_t kShapeIdx = 3;

 private:
  TensorPtr indptr_;
  TensorPtr indices_;
  TensorPtr values_;
};
using CSRTensorPtr = std::shared_ptr<CSRTensor>;

// COOTensor entity class
class MS_CORE_API COOTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create COOTensor with given data type from another tensor.
  ///
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the COOTensor.
  COOTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
      : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

  /// Destructor of COOTensor.
  ~COOTensor() override = default;

  MS_DECLARE_PARENT(COOTensor, MetaSparseTensor)

  /// \brief Gets COOTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets COOTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  TensorPtr GetTensorAt(size_t index) const;

  const size_t GetTensorLength() const { return kShapeIdx + shape().size(); }

  /// \brief Compare two cootensor objects to see if they have same address.
  ///
  /// \param[in] coo_tensor The cootensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const COOTensor &coo_tensor) const { return &coo_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<COOTensor>()) {
      auto &other_ = static_cast<const COOTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  static constexpr size_t kIndicesIdx = 0;
  static constexpr size_t kValuesIdx = 1;
  static constexpr size_t kShapeIdx = 2;

 private:
  TensorPtr indices_;
  TensorPtr values_;
};
using COOTensorPtr = std::shared_ptr<COOTensor>;

// RowTensor entity class
class MS_CORE_API RowTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create RowTensor with given data type from another tensor.
  ///
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the RowTensor.
  RowTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
      : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

  /// Destructor of RowTensor.
  ~RowTensor() override = default;

  /// \brief Gets RowTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets RowTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  /// \brief Compare two rowtensor objects to see if they have same address.
  ///
  /// \param[in] coo_tensor The rowtensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const RowTensor &row_tensor) const { return &row_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<RowTensor>()) {
      auto &other_ = static_cast<const RowTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

 private:
  TensorPtr indices_;
  TensorPtr values_;
};

// Convert shape vector to string.
MS_CORE_API std::string ShapeToString(const ShapeVector &shape);

inline static void CopyTensorData(const TensorDataPtr &dest, const TensorDataPtr &src) {
  auto dest_bytes = dest->nbytes();
  auto src_bytes = src->nbytes();
  auto err = common::huge_memcpy(static_cast<uint8_t *>(dest->data()), dest_bytes,
                                 static_cast<const uint8_t *>(src->const_data()), src_bytes);
  if (err != EOK) {
    MS_LOG(INTERNAL_EXCEPTION) << "Copy tensor data failed! bytes: " << dest_bytes << "/" << src_bytes << ".";
  }
}

using RowTensorPtr = std::shared_ptr<RowTensor>;
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSOR_H_
