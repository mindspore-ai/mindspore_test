/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "frontend/ir/tensor_py.h"

#include <utility>
#include <algorithm>
#include <map>

#include "pybind11/complex.h"

#include "include/common/pybind_api/api_register.h"
#include "abstract/abstract_value.h"
#include "utils/cache_embedding_hashmap_struct.h"
#include "include/common/utils/python_adapter.h"
#include "mindspore/ccsrc/include/backend/distributed/embedding_cache/embedding_cache_utils.h"
#include "debug/profiler/profiler.h"
#include "include/common/pynative/adapter.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pipeline/pipeline.h"
#include "include/backend/mbuf_device_address.h"
#include "utils/ordered_set.h"
#include "runtime/device/move_to.h"

namespace mindspore {
namespace tensor {
namespace {
struct TensorToNumpyRegister {
  TensorToNumpyRegister() { python_adapter::PyAdapterCallback::SetTensorToNumpyHandler(tensor::TensorPybind::AsNumpy); }
} callback_register;
}  // namespace
constexpr ssize_t kPyBufItemSize1 = 1;
constexpr ssize_t kPyBufItemSize2 = 2;
constexpr ssize_t kPyBufItemSize4 = 4;
constexpr ssize_t kPyBufItemSize8 = 8;

static TypeId GetDataType(const py::buffer_info &buf) {
  if (buf.format.size() == 1) {
    switch (buf.format.front()) {
      case 'e':
      case 'f':
      case 'd':
        switch (buf.itemsize) {
          case kPyBufItemSize2:
            return TypeId::kNumberTypeFloat16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeFloat32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeFloat64;
        }
        break;
      case 'b':
      case 'h':
      case 'i':
      case 'l':
      case 'q':
        switch (buf.itemsize) {
          case kPyBufItemSize1:
            return TypeId::kNumberTypeInt8;
          case kPyBufItemSize2:
            return TypeId::kNumberTypeInt16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeInt32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeInt64;
          default:
            break;
        }
        break;
      case 'B':
      case 'H':
      case 'I':
      case 'L':
      case 'Q':
        switch (buf.itemsize) {
          case kPyBufItemSize1:
            return TypeId::kNumberTypeUInt8;
          case kPyBufItemSize2:
            return TypeId::kNumberTypeUInt16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeUInt32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeUInt64;
          default:
            break;
        }
        break;
      case '?':
        return TypeId::kNumberTypeBool;
      case 'E':
        return TypeId::kNumberTypeBFloat16;
      default:
        break;
    }
  } else if (buf.format.size() >= 2) {
    // Support np.str_ dtype, format: {x}w. {x} is a number that means the maximum length of the string items.
    if (buf.format.back() == 'w' || buf.format.back() == 's') {
      return TypeId::kObjectTypeString;
    } else if (buf.format == "Zf") {
      return TypeId::kNumberTypeComplex64;
    } else if (buf.format == "Zd") {
      return TypeId::kNumberTypeComplex128;
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << ", item size " << buf.itemsize;
  return TypeId::kTypeUnknown;
}

static std::string GetPyTypeFormat(TypeId data_type) {
  switch (data_type) {
    case TypeId::kNumberTypeFloat16:
      return "e";
    case TypeId::kNumberTypeBFloat16:
      return "E";
    case TypeId::kNumberTypeFloat32:
      return py::format_descriptor<float>::format();
    case TypeId::kNumberTypeFloat64:
      return py::format_descriptor<double>::format();
    case TypeId::kNumberTypeUInt8:
      return py::format_descriptor<uint8_t>::format();
    case TypeId::kNumberTypeUInt16:
      return py::format_descriptor<uint16_t>::format();
    case TypeId::kNumberTypeUInt32:
      return py::format_descriptor<uint32_t>::format();
    case TypeId::kNumberTypeUInt64:
      return py::format_descriptor<uint64_t>::format();
    case TypeId::kNumberTypeInt4:
    case TypeId::kNumberTypeInt8:
      return py::format_descriptor<int8_t>::format();
    case TypeId::kNumberTypeInt16:
      return py::format_descriptor<int16_t>::format();
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt32:
      return py::format_descriptor<int32_t>::format();
    case TypeId::kNumberTypeInt64:
      return py::format_descriptor<int64_t>::format();
    case TypeId::kNumberTypeBool:
      return py::format_descriptor<bool>::format();
    case TypeId::kObjectTypeString:
      return py::format_descriptor<uint8_t>::format();
    case TypeId::kNumberTypeComplex64:
      return py::format_descriptor<std::complex<float>>::format();
    case TypeId::kNumberTypeComplex128:
      return py::format_descriptor<std::complex<double>>::format();
    case TypeId::kMetaTypeType:
    case TypeId::kMetaTypeEllipsis:
    default:
      MS_LOG(WARNING) << "Unsupported DataType " << data_type << ".";
      return "";
  }
}

static bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & static_cast<unsigned int>(pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) != 0;
}

// TensorDataNumpy implements TensorData using numpy array.
class TensorDataNumpy : public TensorData {
 public:
  explicit TensorDataNumpy(py::buffer_info &&buffer) : buffer_(std::make_unique<py::buffer_info>(std::move(buffer))) {}

  ~TensorDataNumpy() override {
    py::gil_scoped_acquire acquire;
    buffer_.reset();
  }

  /// Total number of elements.
  ssize_t size() const override { return buffer()->size; }

  /// Byte size of a single element.
  ssize_t itemsize() const override { return buffer()->itemsize; }

  /// Total number of bytes.
  ssize_t nbytes() const override { return buffer()->itemsize * buffer()->size; }

  /// Number of dimensions.
  ssize_t ndim() const override { return buffer()->ndim; }

  /// Data pointer.
  void *data() override { return buffer_data(); }

  const void *const_data() const override { return buffer()->ptr; }

  bool is_sub_data() const override { return false; }

  bool has_sub_data() const override { return false; }

  bool is_from_numpy() const override { return true; }

  const std::vector<ssize_t> &shape() const { return buffer()->shape; }

  /// To string.
  std::string ToString(const TypeId, const ShapeVector &, bool use_comma) const override {
    py::gil_scoped_acquire gil_acquire;
    if (use_comma) {
      // Call python np.array2string(data_, separator=', ') to convert string with comma.
      py::dict kwargs;
      kwargs["separator"] = ", ";
      auto np = py::module::import("numpy");
      auto array2string = np.attr("array2string");
      return py::str(array2string(py_array(), **kwargs));
    }
    // without comma.
    return py::str(py_array());
  }

  /// py::array object. by default, use py::str() as the dummy owner to prevent data copy.
  py::array py_array(const py::handle &owner = py::str()) const {
    py::gil_scoped_acquire acquire;
    py::dtype np_dtype =
      (buffer()->format == "E") ? py::detail::npy_format_descriptor<bfloat16>::dtype() : py::dtype(*buffer());
    return py::array(np_dtype, buffer()->shape, buffer()->strides, buffer()->ptr, owner);
  }

 private:
  void *buffer_data() const { return buffer_->ptr; }
  std::unique_ptr<py::buffer_info> const &buffer() const {
    MS_EXCEPTION_IF_NULL(buffer_);
    return buffer_;
  }

  // The internal buffer.
  std::unique_ptr<py::buffer_info> buffer_;
};

// This class is uesd to get huge tensor data from persistent storage. Tensor data can be got by slice.
// It used at extend embedding to persistent storage.
class PersistentTensorDataNumpy : public TensorDataNumpy {
 public:
  explicit PersistentTensorDataNumpy(py::buffer_info &&buffer, int slice_num)
      : TensorDataNumpy(std::move(buffer)), slice_num_(slice_num) {}

  ~PersistentTensorDataNumpy() override = default;

  // Fill data with a special slice tensor data. It will read data from persistent storage.
  void FillSliceData(const int32_t param_key, const int slice_index) {
    if (slice_index >= slice_num_) {
      MS_LOG(ERROR) << "Slice index is out of range, index: " << slice_index;
      return;
    }
    auto emb_store = embedding_storage_manager.Get(param_key);
    MS_EXCEPTION_IF_NULL(emb_store);

    size_t first_dim = (size_t)SliceDataShape()[0];
    size_t start_key = slice_index * first_dim;
    std::vector<int> keys(first_dim);
    std::iota(keys.begin(), keys.end(), start_key);
    if (!emb_store->Get({keys.data(), first_dim * sizeof(int)}, {this->data(), LongToSize(this->nbytes())})) {
      MS_LOG(EXCEPTION) << "Failed to get data from embedding store!";
    }
  }

  const std::vector<ssize_t> &SliceDataShape() const { return this->shape(); }

  // Get total silce num of tensor data.
  int slice_num() const { return slice_num_; }

  bool is_persistent_data() const override { return true; }

 private:
  int slice_num_{1};
};

py::buffer_info TensorPybind::GetPyBufferFromPyArray(const py::array &input) {
  py::buffer_info buf;
  auto descr = py::detail::array_descriptor_proxy(py::detail::array_proxy(input.ptr())->descr);
  // For bfloat16, modify descr->type_num to support acquiring buffer_info from numpy.
  if (descr->type == 'E') {
    // convert descr->type_num from E(NPY_BFLOAT16) to H(NPY_USHORT)
    const int NPY_USHORT = 4;
    int orig_type_num = descr->type_num;
    descr->type_num = NPY_USHORT;
    // acquire buffer_info with type of NPY_USHORT
    buf = input.request();
    // convert buffer_info.format from H(NPY_USHORT) to E(NPY_BFLOAT16)
    buf.format = "E";
    // change back descr->type_num
    descr->type_num = orig_type_num;
  } else {
    buf = input.request();
  }
  return buf;
}

TensorPtr TensorPybind::MakeTensor(const py::array &input, const TypePtr &type_ptr) {
  py::gil_scoped_acquire acquire;
  // Get input buffer info.
  py::buffer_info buf = TensorPybind::GetPyBufferFromPyArray(input);
  // Check data types.
  auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  auto buf_type = GetDataType(buf);
  if (buf_type == TypeId::kTypeUnknown && data_type == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
  }
  MS_LOG(DEBUG) << "data_type: " << data_type << ", buf_type: " << buf_type;
  if (data_type == TypeId::kObjectTypeString || buf_type == TypeId::kObjectTypeString) {
    return TensorPybind::MakeTensorOfNumpy(input);
  }
  // Use buf type as data type if type_ptr not set.
  if (data_type == TypeId::kTypeUnknown) {
    data_type = buf_type;
  }
  // Convert input array to C contiguous if need.
  std::unique_ptr<char[]> tmp_buf;
  if (!IsCContiguous(input)) {
    Py_buffer pybuf;
    if (PyObject_GetBuffer(input.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS) != 0) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }
    tmp_buf = std::make_unique<char[]>(pybuf.len);
    if (PyBuffer_ToContiguous(tmp_buf.get(), &pybuf, pybuf.len, 'C') != 0) {
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    buf.ptr = tmp_buf.get();
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());
  if (data_type == buf_type) {
    // Use memory copy if input data type is the same as the required type.
    return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf.size * buf.itemsize);
  }
  // Create tensor with data type converted.
  return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf_type);
}

/// Creates a Tensor from a numpy array without copy
TensorPtr TensorPybind::MakeTensorOfNumpy(const py::array &input) {
  py::gil_scoped_acquire acquire;
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = TensorPybind::GetPyBufferFromPyArray(input);
  // Get tensor dtype and check it.
  auto dtype = GetDataType(buf);
  if (dtype == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());
  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<TensorDataNumpy>(std::move(buf));
  return std::make_shared<Tensor>(dtype, shape, tensor_data);
}

/// Creates a Tensor from a numpy array without copy, use persistent tensor data
TensorPtr TensorPybind::MakePersistentDataTensorOfNumpy(const py::array &input, const py::int_ slice_num) {
  py::gil_scoped_acquire acquire;
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = TensorPybind::GetPyBufferFromPyArray(input);
  // Get tensor dtype and check it.
  auto dtype = GetDataType(buf);
  if (dtype == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());
  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<PersistentTensorDataNumpy>(std::move(buf), static_cast<int>(slice_num));
  return std::make_shared<Tensor>(dtype, shape, tensor_data);
}

void TensorPybind::SetUserData(const BaseTensorPtr &tensor, const py::str &key, const py::object &value) {
  const std::string name = key.cast<std::string>();
  const auto &primitive_data = std::make_shared<TensorPyUserData>();
  primitive_data->obj = value;
  const_cast<BaseTensorPtr &>(tensor)->set_user_data<TensorPyUserData>(name, primitive_data);
}

py::object TensorPybind::GetUserData(const BaseTensorPtr &tensor, const py::str &key) {
  const std::string name = key.cast<std::string>();
  const auto primitive_data = tensor->user_data<TensorPyUserData>(name);
  if (primitive_data == nullptr) {
    return py::none();
  }
  return primitive_data->obj;
}

static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

static py::buffer_info GetPyBufferInfo(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.data().itemsize());
  return py::buffer_info{
    tensor.data_c(), tensor.data().itemsize(), GetPyTypeFormat(tensor.data_type()), tensor.DataDim(), shape, strides};
}

py::tuple TensorPybind::GetPyTupleShape(const Tensor &tensor) {
  auto &shape = tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::tuple TensorPybind::GetPyTupleStrides(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.data().itemsize());
  py::tuple py_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    py_strides[i] = py::int_(strides[i]);
  }
  return py_strides;
}

py::int_ TensorPybind::GetPyItemSize(const Tensor &tensor) { return tensor.data().itemsize(); }

py::int_ TensorPybind::GetPyNBytes(const Tensor &tensor) { return tensor.data().nbytes(); }

template <typename T>
void MemCopyFromCacheToHost(void *hashmap_addr, void *host_addr, void *cache_addr, size_t host_max, size_t cache_max,
                            size_t hashmap_size, size_t col_size) {
  auto host_data = static_cast<char *>(host_addr);
  auto cache_data = static_cast<char *>(cache_addr);
  auto hashmap_data = static_cast<HashmapEntry<T> *>(hashmap_addr);
  // default param type float
  const size_t param_type_size = 4;
  size_t single_col_bytes = param_type_size * col_size;
  for (size_t i = 0; i < hashmap_size; ++i) {
    if (!hashmap_data[i].IsEmpty()) {
      size_t host_offset = single_col_bytes * LongToSize(hashmap_data[i].key_);
      size_t cache_offset = single_col_bytes * LongToSize(hashmap_data[i].value_);
      if (cache_offset + single_col_bytes <= cache_max) {
        auto ret =
          memcpy_s(host_data + host_offset, host_max - host_offset, cache_data + cache_offset, single_col_bytes);
        if (ret != 0) {
          MS_LOG(EXCEPTION) << "Memcpy failed.";
        }
      }
    }
  }
  MS_LOG(INFO) << "Memcpy from cache to host success!";
}

void TensorPybind::FlushFromCache(const Tensor &tensor) {
  py::gil_scoped_release gil_release;
  tensor.data_sync();

  if (tensor.cache_enable()) {
    MS_LOG(INFO) << tensor.ToString() << " is cache enable.";
    auto hashmap_tensor_ptr = tensor.hashmap_tensor_ptr();
    auto cache_tensor_ptr = tensor.cache_tensor_ptr();
    if (hashmap_tensor_ptr != nullptr && cache_tensor_ptr != nullptr) {
      hashmap_tensor_ptr->data_sync();
      cache_tensor_ptr->data_sync();
      auto hashmap_size = hashmap_tensor_ptr->shape_c()[0];
      auto host_shape = tensor.shape_c();
      auto cache_shape = cache_tensor_ptr->shape_c();
      if (host_shape.size() != 2 && cache_shape.size() != 2 && host_shape[1] != cache_shape[1]) {
        MS_LOG(EXCEPTION) << "Got host shape and cache shape invalid."
                          << "host shape:" << host_shape << ", cache shape:" << cache_shape;
      }
      auto host_data_max_size = static_cast<size_t>(tensor.Size());
      auto cache_data_max_size = static_cast<size_t>(cache_tensor_ptr->Size());
      auto hashmap_data_type = hashmap_tensor_ptr->data_type();
      if (hashmap_data_type == TypeId::kNumberTypeInt32) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), tensor.data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else if (hashmap_data_type == TypeId::kNumberTypeInt64) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), tensor.data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else {
        MS_LOG(ERROR) << "Hashmap dtype only suppotr int32, in64.";
      }
    }
  }
}

py::bytes TensorPybind::GetBytes(const Tensor &tensor) {
  py::gil_scoped_acquire acquire;
  if (tensor.get_copy_done_flag()) {
    const_cast<Tensor &>(tensor).set_copy_done_flag(false);
    return py::bytes(static_cast<const char *>(tensor.data_c()), tensor.Size());
  }
  tensor.data_sync();
  return py::bytes(static_cast<const char *>(tensor.data_c()), tensor.Size());
}

void CopyFromBuffer(char *dst, size_t dst_size, const char *src, size_t src_size, TypeId data_type) {
  bool fp16_in_fp32 = (data_type == TypeId::kNumberTypeBFloat16) && (dst_size * 2 == src_size);
  if (fp16_in_fp32) {
    int elem_num = static_cast<int>(src_size / sizeof(float));
    for (int i = 0; i < elem_num; ++i) {
      auto dst_ptr = static_cast<char *>(dst + i * sizeof(bfloat16));
      auto src_ptr = static_cast<const char *>(src + sizeof(bfloat16) + i * sizeof(float));
      errno_t ret = memcpy_s(dst_ptr, sizeof(bfloat16), src_ptr, sizeof(bfloat16));
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor:" << ret;
      }
    }
  } else {
    size_t remain_size = src_size;
    auto dst_ptr = dst;
    auto src_ptr = src;
    while (remain_size > SECUREC_MEM_MAX_LEN) {
      auto ret = memcpy_s(dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor" << ret;
      }
      remain_size -= SECUREC_MEM_MAX_LEN;
      dst_ptr += SECUREC_MEM_MAX_LEN;
      src_ptr += SECUREC_MEM_MAX_LEN;
    }
    if (remain_size != 0U) {
      auto ret = memcpy_s(dst_ptr, remain_size, src_ptr, remain_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor" << ret;
      }
    }
  }
}

TensorPtr TensorPybind::ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                             const TypePtr &type_ptr) {
  ShapeVector shape;
  for (size_t i = 0; i < dims.size(); ++i) {
    shape.push_back(dims[i].cast<int>());
  }
  TypeId data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(data_type, shape);
  const char *tensor_buf = PYBIND11_BYTES_AS_STRING(bytes_obj.ptr());
  char *tensor_data_buf = reinterpret_cast<char *>(tensor->data_c());
  CopyFromBuffer(tensor_data_buf, tensor->Size(), tensor_buf, PYBIND11_BYTES_SIZE(bytes_obj.ptr()), data_type);
  return tensor;
}

py::object GetItemForToList(void *data, const TypeId &data_type, const int &index) {
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return py::int_(py::cast(*(static_cast<int8_t *>(data) + index)));
    case TypeId::kNumberTypeUInt8:
      return py::int_(py::cast(*(static_cast<uint8_t *>(data) + index)));
    case TypeId::kNumberTypeInt16:
      return py::int_(py::cast(*(static_cast<int16_t *>(data) + index)));
    case TypeId::kNumberTypeUInt16:
      return py::int_(py::cast(*(static_cast<uint16_t *>(data) + index)));
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt32:
      return py::int_(py::cast(*(static_cast<int *>(data) + index)));
    case TypeId::kNumberTypeUInt32:
      return py::int_(py::cast(*(static_cast<uint32_t *>(data) + index)));
    case TypeId::kNumberTypeInt64:
      return py::int_(py::cast(*(static_cast<int64_t *>(data) + index)));
    case TypeId::kNumberTypeUInt64:
      return py::int_(py::cast(*(static_cast<uint64_t *>(data) + index)));
    case TypeId::kNumberTypeFloat16:
      return py::float_(py::cast(*(static_cast<float16 *>(data) + index)));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return py::float_(py::cast(*(static_cast<float *>(data) + index)));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return py::float_(py::cast(*(static_cast<double *>(data) + index)));
    case TypeId::kNumberTypeBFloat16:
      return py::float_(py::cast(*(static_cast<bfloat16 *>(data) + index)));
    case TypeId::kNumberTypeBool:
      return py::bool_(py::cast(*(static_cast<bool *>(data) + index)));
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex:
      return py::cast(
        std::complex<double>{*(static_cast<float *>(data) + index * 2), *(static_cast<float *>(data) + 1 + index * 2)});
    case TypeId::kNumberTypeComplex128:
      return py::cast(std::complex<long double>{*(static_cast<double *>(data) + index * 2),
                                                *(static_cast<double *>(data) + 1 + index * 2)});
    default:
      MS_EXCEPTION(TypeError) << "Not support tensor data type: " << data_type << ".";
      break;
  }
  return py::none();
}

py::object RecursiveToList(void *data, const size_t &size, const std::vector<int64_t> &stride,
                           const std::vector<int64_t> &shape, const size_t &element_size, const TypeId &data_type,
                           int cur_dim, int index) {
  int ndim = shape.size();
  py::list res_list;
  if (size == 0 && cur_dim + 1 == ndim) {
    return res_list;
  }
  if (cur_dim == ndim) {
    return GetItemForToList(data, data_type, index);
  }
  for (int i = 0; i < shape[cur_dim]; ++i) {
    py::object obj = RecursiveToList(data, ndim, stride, shape, element_size, data_type, cur_dim + 1, index);
    res_list.append(obj);
    index += stride[cur_dim];
  }
  return res_list;
}

py::object TensorPybind::ToList(const BaseTensorPtr &tensor) {
  tensor->data_sync();
  auto tensor_element_count = tensor->data().size();
  auto tensor_stride = tensor->stride();
  auto tensor_shape = tensor->shape();
  auto element_size = tensor->data().itemsize();
  auto data = tensor->data_c();
  auto data_type = tensor->data_type();

  return RecursiveToList(data, tensor_element_count, tensor_stride, tensor_shape, element_size, data_type, 0, 0);
}

py::object TensorPybind::Item(const BaseTensorPtr &tensor) {
  auto tensor_element_count = tensor->data().size();
  if (tensor_element_count != 1) {
    MS_EXCEPTION(ValueError) << "The tensor should have only one element, but got " << tensor_element_count << ","
                             << " more than one element is ambiguous.";
  }
  tensor->data_sync();
  auto data_type = tensor->data_type();
  auto data = tensor->data_c();
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return py::int_(py::cast(*static_cast<int8_t *>(data)));
    case TypeId::kNumberTypeUInt8:
      return py::int_(py::cast(*static_cast<uint8_t *>(data)));
    case TypeId::kNumberTypeInt16:
      return py::int_(py::cast(*static_cast<int16_t *>(data)));
    case TypeId::kNumberTypeUInt16:
      return py::int_(py::cast(*static_cast<uint16_t *>(data)));
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt32:
      return py::int_(py::cast(*static_cast<int *>(data)));
    case TypeId::kNumberTypeUInt32:
      return py::int_(py::cast(*static_cast<uint32_t *>(data)));
    case TypeId::kNumberTypeInt64:
      return py::int_(py::cast(*static_cast<int64_t *>(data)));
    case TypeId::kNumberTypeUInt64:
      return py::int_(py::cast(*static_cast<uint64_t *>(data)));
    case TypeId::kNumberTypeFloat16:
      return py::float_(py::cast(*static_cast<float16 *>(data)));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return py::float_(py::cast(*static_cast<float *>(data)));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return py::float_(py::cast(*static_cast<double *>(data)));
    case TypeId::kNumberTypeBFloat16:
      return py::float_(py::cast(*static_cast<bfloat16 *>(data)));
    case TypeId::kNumberTypeBool:
      return py::bool_(py::cast(*static_cast<bool *>(data)));
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex:
      return py::cast(std::complex<double>{(*static_cast<float *>(data)), (*(static_cast<float *>(data) + 1))});
    case TypeId::kNumberTypeComplex128:
      return py::cast(std::complex<long double>{(*static_cast<double *>(data)), (*(static_cast<double *>(data) + 1))});
    default:
      MS_EXCEPTION(TypeError) << "Not support tensor data type: " << data_type << ".";
      break;
  }
  return py::none();
}

py::array TensorPybind::SyncAsNumpy(const Tensor &tensor) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kAsnumpy);
  // Asnumpy should be a read-only operation and should not modify the original Tensor.
  if (tensor.need_pipeline_sync()) {
    runtime::Pipeline::Get().WaitAll();
  }
  Tensor tensor_for_copy(tensor);
  {
    py::gil_scoped_release gil_release;

    // BFloat16 may not be supported in numpy.
    std::string numpy_version = np_dtypes::GetNumpyVersion();
    std::string minimum_numpy_version = np_dtypes::GetMinimumSupportedNumpyVersion();
    if (tensor.data_type() == kNumberTypeBFloat16 && !np_dtypes::NumpyVersionValid(numpy_version)) {
      MS_EXCEPTION(TypeError) << "For asnumpy, the numpy bfloat16 data type is supported in Numpy versions "
                              << minimum_numpy_version << " to " << minimum_numpy_version[0] << ".x.x, but got "
                              << numpy_version << ", please upgrade numpy version.";
    }

    // To be deleted
    if (!tensor.get_copy_done_flag()) {
      tensor_for_copy.data_sync();
    }
    const_cast<Tensor &>(tensor).set_copy_done_flag(false);

    // To be deleted
    // Release device address of graph output tensor.
    if (tensor.need_release_device_mem()) {
      const_cast<Tensor &>(tensor).set_device_address(nullptr);
    }
  }
  return AsNumpy(tensor_for_copy);
}

py::array TensorPybind::AsNumpy(const Tensor &tensor) {
  // Use TensorData as the owner to prevent use-after-free problem.
  // We can NOT use Tensor as the owner since its TensorData may change
  // by other operations such as AssignValue().
  py::gil_scoped_acquire acquire;
  py::object owner = py::cast(tensor.data_ptr());
  auto data_numpy = dynamic_cast<const TensorDataNumpy *>(&tensor.data());
  if (data_numpy != nullptr) {
    // Return internal numpy array if tensor data is implemented base on it.
    return data_numpy->py_array(owner);
  }
  // Otherwise, create numpy array by buffer protocol.
  auto info = GetPyBufferInfo(tensor);
  py::dtype np_dtype = (tensor.data_type() == kNumberTypeBFloat16)
                         ? py::detail::npy_format_descriptor<bfloat16>::dtype()
                         : py::dtype(info);
  return py::array(np_dtype, info.shape, info.strides, info.ptr, owner);
}

void TensorPybind::Offload(const BaseTensorPtr &tensor, bool release) {
  py::gil_scoped_release gil_release;
  if (release) {
    const auto &device_sync = tensor->device_address();
    if (device_sync == nullptr) {
      return;
    }
    const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    if (device_address == nullptr) {
      return;
    }
    if (device_address->GetPtr() == nullptr) {
      MS_LOG(INFO) << "For Offload, this tensor's device_ptr is nullptr, it may have been offloaded or released by"
                   << " the framework.";
      return;
    }
    MS_LOG(INFO) << "Tensor Offload start, the tensor's device_address is : " << device_address.get()
                 << ", the tensor's size is : " << device_address->GetSize();
    device_address->SyncDeviceToHost(device_address->GetSize(), tensor->data_c());
    device_address->ClearDeviceMemory();
  } else {
    tensor->data_sync();
    // Release device address of graph output tensor.
    const_cast<BaseTensorPtr &>(tensor)->set_device_address(nullptr);
  }
}

void TensorPybind::Load(const Tensor &tensor) {
  py::gil_scoped_release gil_release;
  const auto &device_sync = tensor.device_address();
  if (device_sync == nullptr) {
    MS_LOG(WARNING) << "Tensor has no DeviceSync, can not be loaded.";
    return;
  }
  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  if (device_address == nullptr) {
    MS_LOG(WARNING) << "Tensor has no DeviceAddress, can not be loaded.";
    return;
  }
  if (tensor.data_c() == nullptr) {
    MS_LOG(WARNING) << "Tensor has no cpu data, can not be loaded.";
    return;
  }
  const auto device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  device_ctx->device_res_manager_->AllocateMemory(device_address.get());
  MS_LOG(INFO) << "Tensor Load start, the tensor's device_address is : " << device_address.get()
               << ", the tensor's size is : " << device_address->GetSize();
  device_address->SyncHostToDevice(device_address->GetSize(), tensor.data_c());
}

void TensorPybind::SetDeviceAddress(const BaseTensorPtr &tensor, uintptr_t addr, const ShapeVector &shape,
                                    const TypePtr type_ptr) {
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    MS_LOG(EXCEPTION) << "set_device_address now only support Ascend backend!";
  }
  uint32_t device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  if (type_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Dtype to be set is nullptr.";
  }

  TypeId data_type = type_ptr->type_id();
  if (data_type != tensor->data_type()) {
    MS_LOG(EXCEPTION) << "Dtype to be set is not euqal with the tensor's, then tensor's dtype is"
                      << tensor->data_type();
  }

  if (shape != tensor->shape()) {
    MS_LOG(EXCEPTION) << "Shape to be set is not euqal with the tensor's, then tensor's shape is" << tensor->shape();
  }

  void *data = reinterpret_cast<void *>(addr);
  size_t elem_num = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    elem_num *= shape[i];
  }
  auto data_size = elem_num * GetDataTypeSize(data_type);
  auto device_sync_ = tensor->device_address();
  if (device_sync_ == nullptr) {
    auto device_address =
      std::make_shared<device::MbufDeviceAddress>(data, data_size, shape, data_type, kAscendDevice, device_id);
    const_cast<BaseTensorPtr &>(tensor)->set_device_address(device_address);
  } else {
    auto device_address = std::dynamic_pointer_cast<device::MbufDeviceAddress>(device_sync_);
    device_address->SetData(data);
  }
}

uintptr_t TensorPybind::DataPtr(const BaseTensorPtr &tensor) {
  runtime::Pipeline::Get().WaitForward();
  const auto device_address = tensor->device_address();
  if (device_address == nullptr) {
    MS_LOG(ERROR) << "Tensor device address is null";
    return reinterpret_cast<uintptr_t>(nullptr);
  }
  auto *data_ptr = device_address->GetMutablePtr();
  MS_LOG(DEBUG) << "Get Tensor data ptr " << data_ptr;
  return reinterpret_cast<uintptr_t>(data_ptr);
}

TensorPtr TensorPybind::MoveTo(const Tensor &self, const std::string &to, bool blocking) {
  py::gil_scoped_release gil_release;
  MS_LOG(INFO) << "Try move tensor to " << to;
  auto target_tensor = std::make_shared<tensor::Tensor>(self.data_type(), self.shape());
  target_tensor->set_device_address(nullptr);
  bool return_self = false;
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  device::MoveTo(std::make_shared<tensor::Tensor>(self), target_tensor, to, blocking, &return_self);
  if (return_self) {
    return std::make_shared<tensor::Tensor>(self);
  }
  return target_tensor;
}

py::array TensorPybind::AsNumpyOfSlice(const Tensor &tensor, const int32_t param_key, const int slice_index) {
  py::gil_scoped_acquire acquire;
  py::object owner = py::cast(tensor.data_ptr());
  auto data_numpy = std::dynamic_pointer_cast<PersistentTensorDataNumpy>(tensor.data_ptr());
  MS_EXCEPTION_IF_NULL(data_numpy);

  data_numpy->FillSliceData(param_key, slice_index);

  // Return internal numpy array if tensor data is implemented base on it.
  // And persistent tensor data is only implemented base on numpy array.
  return data_numpy->py_array(owner);
}

py::object TensorPybind::TensorGetItem(const py::object &self, const py::object &py_index) {
  static std::string config_static_shape = common::GetEnv("MS_PYNATIVE_CONFIG_STATIC_SHAPE");
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice ||
      MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode || config_static_shape == "1") {
    return self.attr("_getitem_origin")(py_index);
  }
  return self.attr("_getitem")(py_index);
}

py::object TensorPybind::TensorSetItem(const py::object &self, const py::object &py_index, const py::object &py_value) {
  static std::string config_static_shape = common::GetEnv("MS_PYNATIVE_CONFIG_STATIC_SHAPE");
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice ||
      MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode || config_static_shape == "1") {
    return self.attr("_setitem_origin")(py_index, py_value);
  }
  return self.attr("_setitem")(py_index, py_value);
}

py::object TensorPyImpl::GetInitializerFromPython(const py::dict &input) {
  if (!input.contains("init") || py::isinstance<py::none>(input["init"])) {
    return py::none();
  }
  return input["init"];
}

bool TensorPyImpl::GetConstArgFromPython(const py::dict &input) {
  if (!input.contains("const_arg") || py::isinstance<py::none>(input["const_arg"])) {
    return false;
  }
  py::object obj = input["const_arg"];
  if (!PyBool_Check(obj.ptr())) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the type of 'const_arg' should be 'bool', but got type '"
                            << obj.get_type() << "'.";
  }
  return obj.cast<bool>();
}

std::string TensorPyImpl::GetDeviceFromPython(const py::dict &input) {
  if (!input.contains("device") || py::isinstance<py::none>(input["device"])) {
    return "";
  }
  py::object obj = input["device"];
  if (!py::isinstance<py::str>(obj)) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the device should be string, but got '" << obj.get_type() << "'.";
  }
  std::string device = py::str(obj);
  if (std::strncmp(device.c_str(), "CPU", std::strlen("CPU")) != 0) {
    MS_EXCEPTION(ValueError) << "Only 'CPU' is supported for device, but got '" << device << "'.";
  }
  return device;
}

py::object TensorPyImpl::GetSymbolicShapeFromPython(const py::dict &input) {
  if (!input.contains("symbolic_shape")) {
    return py::none();
  }
  py::object obj = input["symbolic_shape"];
  if (py::isinstance<py::none>(obj) || !py::isinstance<py::list>(obj)) {
    return py::none();
  }
  py::list obj_list = py::cast<py::list>(obj);
  if (obj_list.empty()) {
    return py::none();
  }
  return obj;
}

const TypePtr TensorPyImpl::GetDtypeFromPython(const py::dict &input) {
  if (!input.contains("dtype")) {
    return nullptr;
  }
  py::object obj = input["dtype"];
  if (py::isinstance<py::none>(obj)) {
    return nullptr;
  }
  if (!py::isinstance<Type>(obj)) {
    MS_EXCEPTION(TypeError)
      << "For 'Tensor', the 'dtype' should be one of [mindspore.int8, mindspore.int16, mindspore.int32, "
      << "mindspore.int64, mindspore.uint8, mindspore.uint16, mindspore.uint32, mindspore.uint64, mindspore.float16, "
      << "mindspore.float32, mindspore.float64, mindspore.bfloat16, mindspore.complex64, mindspore.complex128, "
      << "mindspore.int4, mindspore.bool_, mindspore.string_], but got '" << obj.get_type() << "'.";
  }
  return obj.cast<TypePtr>();
}

const ShapeVector TensorPyImpl::GetShapeFromPython(const py::dict &input) {
  ShapeVector shape;
  if (!input.contains("shape")) {
    return shape;
  }
  py::object obj = input["shape"];
  if (!py::isinstance<ShapeVector>(obj)) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the 'shape' should be one of [list, tuple], but got '" << obj.get_type()
                            << "'.";
  }
  shape = obj.cast<ShapeVector>();
  return shape;
}

TensorPtr TensorPyImpl::InitTensorByInputDta(const py::dict &input, const TypePtr &dtype) {
  py::object input_obj = input["input_data"];
  if (IsTensorPy(input_obj)) {
    TensorPtr obj = ConvertToTensor(input_obj);
    TypeId data_type = dtype != nullptr ? dtype->type_id() : kTypeUnknown;
    if (data_type == kTypeUnknown || obj->data_type() == data_type) {
      return std::make_shared<Tensor>(*obj);
    }
    return std::make_shared<Tensor>(*obj, data_type);
  }

  if (py::isinstance<py::float_>(input_obj) || py::isinstance<py::int_>(input_obj) ||
      py::isinstance<py::list>(input_obj) || py::isinstance<py::tuple>(input_obj) ||
      PyComplex_CheckExact(input_obj.ptr()) || py::isinstance<py::bytes>(input_obj)) {
    return TensorPybind::MakeTensor(py::array(input_obj), dtype);
  }

  if (py::isinstance<py::array>(input_obj)) {
    return TensorPybind::MakeTensor(input_obj, dtype);
  }

  return nullptr;
}

TensorPtr TensorPyImpl::InitTensorByShape(const py::dict &input, const TypePtr &dtype) {
  if (input.contains("shape") &&
      (py::isinstance<py::list>(input["shape"]) || py::isinstance<py::tuple>(input["shape"]))) {
    TypeId data_type = dtype != nullptr ? dtype->type_id() : TypeId::kNumberTypeFloat64;
    return std::make_shared<Tensor>(data_type, GetShapeFromTuple(input["shape"]));
  }
  ShapeVector shape = GetShapeFromPython(input);
  TypeId data_type = dtype != nullptr ? dtype->type_id() : kTypeUnknown;
  return std::make_shared<Tensor>(data_type, shape);
}

TensorPtr TensorPyImpl::InitTensor(const py::dict &input) {
  TypePtr dtype = GetDtypeFromPython(input);
  TensorPtr output = nullptr;
  if (input.contains("input_data") && (!py::isinstance<py::none>(input["input_data"]))) {
    output = InitTensorByInputDta(input, dtype);
  } else {
    output = InitTensorByShape(input, dtype);
  }
  MS_EXCEPTION_IF_NULL(output);
  return output;
}

const TensorPyPtr TensorPyImpl::InitTensorPy(const py::dict &input) {
  TensorPtr tensor = InitTensor(input);
  TensorPyPtr tensorpy = std::make_shared<TensorPy>(tensor);
  tensorpy->SetInitializer(GetInitializerFromPython(input));
  tensorpy->SetConstArg(GetConstArgFromPython(input));
  tensorpy->SetDevice(GetDeviceFromPython(input));
  tensorpy->SetSymbolicShape(GetSymbolicShapeFromPython(input));
  tensorpy->SetInitFinished(true);
  return tensorpy;
}

TensorPyPtr TensorPyImpl::MakeTensorOfNumpy(const py::array &input) {
  auto tensor = TensorPybind::MakeTensorOfNumpy(input);
  MS_EXCEPTION_IF_NULL(tensor);
  return std::make_shared<TensorPy>(tensor);
}

TensorPyPtr TensorPyImpl::MakePersistentDataTensorOfNumpy(const py::array &input, const py::int_ slice_num) {
  auto tensor = TensorPybind::MakePersistentDataTensorOfNumpy(input, slice_num);
  MS_EXCEPTION_IF_NULL(tensor);
  return std::make_shared<TensorPy>(tensor);
}

TensorPyPtr TensorPyImpl::ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                               const TypePtr &type_ptr) {
  auto tensor = TensorPybind::ConvertBytesToTensor(bytes_obj, dims, type_ptr);
  MS_EXCEPTION_IF_NULL(tensor);
  return std::make_shared<TensorPy>(tensor);
}

void TensorPyImpl::SetOffload(const TensorPyPtr &tensorpy, bool release) {
  auto tensor = tensorpy->GetBaseTensor();
  TensorPybind::Offload(tensor, release);
}

void TensorPyImpl::SetLoad(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetTensor();
  TensorPybind::Load(*tensor);
}

py::bytes TensorPyImpl::GetBytes(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetTensor();
  return TensorPybind::GetBytes(*tensor);
}

py::array TensorPyImpl::SyncAsNumpy(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetTensor();
  return TensorPybind::SyncAsNumpy(*tensor);
}

void TensorPyImpl::FlushFromCache(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetTensor();
  return TensorPybind::FlushFromCache(*tensor);
}

py::array TensorPyImpl::AsNumpyOfSlice(const TensorPyPtr &tensorpy, const int32_t param_key, int slice_index) {
  auto tensor = tensorpy->GetTensor();
  return TensorPybind::AsNumpyOfSlice(*tensor, param_key, slice_index);
}

TensorPyPtr TensorPyImpl::MoveTo(const TensorPyPtr &tensorpy, const std::string &to, bool blocking) {
  auto tensor = tensorpy->GetTensor();
  auto to_tensor = TensorPybind::MoveTo(*tensor, to, blocking);
  MS_EXCEPTION_IF_NULL(to_tensor);
  return std::make_shared<TensorPy>(to_tensor);
}

void TensorPyImpl::SetDeviceAddress(const TensorPyPtr &tensorpy, uintptr_t addr, const ShapeVector &shape,
                                    const TypePtr type_ptr) {
  auto tensor = tensorpy->GetBaseTensor();
  TensorPybind::SetDeviceAddress(tensor, addr, shape, type_ptr);
}

void TensorPyImpl::SetUserData(const TensorPyPtr &tensorpy, const py::str &key, const py::object &value) {
  auto tensor = tensorpy->GetBaseTensor();
  TensorPybind::SetUserData(tensor, key, value);
}

const py::object TensorPyImpl::GetUserData(const TensorPyPtr &tensorpy, const py::str &key) {
  auto tensor = tensorpy->GetBaseTensor();
  return TensorPybind::GetUserData(tensor, key);
}

py::object TensorPyImpl::ToList(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetBaseTensor();
  return TensorPybind::ToList(tensor);
}

py::object TensorPyImpl::Item(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetBaseTensor();
  return TensorPybind::Item(tensor);
}

uint64_t TensorPyImpl::RegisterTensorBackwardHook(const TensorPyPtr &tensorpy, const py::function &hook) {
  auto tensor = tensorpy->GetBaseTensor();
  return pynative::HookAdapter::RegisterTensorBackwardHook(tensor, hook);
}

void TensorPyImpl::RemoveTensorBackwardHook(uint64_t handle_id) {
  pynative::HookAdapter::RemoveTensorBackwardHook(handle_id);
}

py::list TensorPyImpl::GetHooks(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetBaseTensor();
  return pynative::HookAdapter::GetHooks(tensor);
}

uintptr_t TensorPyImpl::DataPtr(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetBaseTensor();
  return TensorPybind::DataPtr(tensor);
}
ShapeVector TensorPyImpl::GetShapeFromTuple(const py::tuple &tuple) {
  ShapeVector shape;
  const size_t size = tuple.size();
  shape.reserve(tuple.size());
  for (size_t i = 0; i < size; ++i) {
    shape.push_back(py::int_(tuple[i]));
  }
  return shape;
}

template <typename T>
py::tuple GetSparseTensorShape(const T &sparse_tensor) {
  auto &shape = sparse_tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::tuple CSRTensorPy::GetPyTupleShape(const CSRTensor &csr_tensor) { return GetSparseTensorShape(csr_tensor); }

py::object CSRTensorPy::GetIndptr(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetIndptr());
}

py::object CSRTensorPy::GetIndices(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetIndices());
}

py::object CSRTensorPy::GetValues(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetValues());
}

py::tuple COOTensorPy::GetPyTupleShape(const COOTensor &coo_tensor) { return GetSparseTensorShape(coo_tensor); }

py::object COOTensorPy::GetIndices(const COOTensorPtr &coo_tensor) {
  return PackTensorToPyObject(coo_tensor->GetIndices());
}

py::object COOTensorPy::GetValues(const COOTensorPtr &coo_tensor) {
  return PackTensorToPyObject(coo_tensor->GetValues());
}

py::tuple RowTensorPy::GetPyTupleShape(const RowTensor &row_tensor) { return GetSparseTensorShape(row_tensor); }

py::object RowTensorPy::GetIndices(const RowTensorPtr &row_tensor) {
  return PackTensorToPyObject(row_tensor->GetIndices());
}

py::object RowTensorPy::GetValues(const RowTensorPtr &row_tensor) {
  return PackTensorToPyObject(row_tensor->GetValues());
}
}  // namespace tensor
}  // namespace mindspore
