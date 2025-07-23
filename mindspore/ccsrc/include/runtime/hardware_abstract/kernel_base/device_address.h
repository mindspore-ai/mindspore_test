/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_DEVICE_TENSOR_H
#define MINDSPORE_DEVICE_TENSOR_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <optional>
#include "ir/tensor.h"
#include "ir/dtype.h"
#include "ir/device_sync.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "ir/tensor_data.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan;
class CPUMemoryManager;
class CPUDeviceContext;
}  // namespace cpu
namespace ascend {
class AscendRuntimeCore;
class AscendMemoryManager;
class AscendResManager;
class DataDumper;
namespace tasksink {
class TaskGenerator;
}  // namespace tasksink
}  // namespace ascend
namespace gpu {
class GPUMemoryManager;
class GPUDeviceContext;
class GPUResManager;
}  // namespace gpu
}  // namespace device
class SingleOpInferSession;
class RuntimeUtils;
}  // namespace mindspore

namespace mindspore {
class AddressAllocator {
 public:
  /**
   * @brief Allocate memory for device address
   * @param size - The size of memory that needs to be allocated
   * @param stream_id - Stream ID for memory allocation
   * @return Raw pointer to the allocated memory
   */
  virtual void *Alloc(size_t size, uint32_t stream_id) = 0;

  /**
   * @brief Free memory for device address
   * @param address_ptr - Raw pointer in DevicePointer that needs to be freed
   * @return true if free succeeds, false otherwise
   */
  virtual bool Free(void *address_ptr) = 0;
};

// DevicePointer encapsulates pointer and reference count-related operations, and supports custom allocator and
// delteter resources. In Ref scenarios, KernelTensor of different DeviceAddress may hold the same DevicePointer
// object.
class DevicePointer {
 public:
  // The arguments are pointer and a bool variable that identifies whether pointer is from the memory pool.
  using Deleter = std::function<void(void *, bool)>;

  DevicePointer() = default;
  explicit DevicePointer(void *ptr) : ptr_(ptr) {}
  DevicePointer(void *ptr, const Deleter &deleter, std::shared_ptr<AddressAllocator> allocator = nullptr)
      : ptr_(ptr), deleter_(deleter), allocator_(std::move(allocator)) {}

  DevicePointer(const DevicePointer &) = delete;
  DevicePointer &operator=(const DevicePointer &) = delete;

  ~DevicePointer() {
    try {
      if (ptr_ != nullptr && allocator_ && from_mem_pool_) {
        allocator_->Free(ptr_);
      } else if (ptr_ != nullptr && deleter_) {
        deleter_(ptr_, from_mem_pool_);
      }
      ptr_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "DevicePointer destructed failed: " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "DevicePointer destructed failed.";
    }
  }

  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " ptr:" << ptr_ << " from mem pool:" << from_mem_pool_ << " deleter:" << (deleter_ != nullptr);
    return ofs.str();
  }

  // Get raw pointer.
  void *ptr() const { return ptr_; }
  // Set raw pointer.
  void set_ptr(void *ptr) { ptr_ = ptr; }

  // Get whether pointer in DevicePointer is allocated from the memory pool.
  bool from_mem_pool() const { return from_mem_pool_; }
  // Set whether pointer in DevicePointer is allocated from the memory pool.
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }

  // Get pointer resource destructor.
  Deleter deleter() const { return deleter_; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { deleter_ = deleter; }

  std::shared_ptr<AddressAllocator> allocator() const { return allocator_; }

  void set_allocator(std::shared_ptr<AddressAllocator> allocator) { allocator_ = allocator; }

 private:
  void *ptr_{nullptr};

  // Whether ptr_  is allocated from the memory pool.
  bool from_mem_pool_{false};

  // The pointer resource destructor.
  Deleter deleter_;

  // The device address allocator that contains allocate memory and delete memory functions.
  std::shared_ptr<AddressAllocator> allocator_;
};
using DevicePointerPtr = std::shared_ptr<DevicePointer>;

enum class NeedAllocateHeteRes : int64_t { NoNeedHeteRes = 0, NeedHostMem = 1, NeedDiskFile = 2 };
struct HeterogeneousInfo {
  // Address on cpu ddr when the KernelTensor is stored on CPU.
  void *host_ptr_;
  // File name when the KernelTensor is stored on Disk.
  std::string file_name_;
  // Token for unfinished async io.
  std::optional<size_t> aio_token_;
  // Mark which heterogeneous resource should be allocated.
  NeedAllocateHeteRes need_alloc_hete_res_{NeedAllocateHeteRes::NoNeedHeteRes};
  std::string ToString() {
    std::ostringstream ofs;
    ofs << this << " host ptr:" << host_ptr_ << " file name:" << file_name_
        << " need alloc hete res:" << need_alloc_hete_res_;
    return ofs.str();
  }
};
using HeterogeneousInfoPtr = std::shared_ptr<HeterogeneousInfo>;
namespace device {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
using TensorPtr = std::shared_ptr<tensor::Tensor>;

enum class StorageType { kDevice, kHost, kFile };

// The flag of device address.
constexpr size_t kDeviceAddressFlagInit = 0;
// Indicates that it is the device address of ref node.
constexpr size_t kDeviceAddressFlagRefNode = 1;
// Indicates that it is the device address of node which has no user.
constexpr size_t kDeviceAddressFlagNotUsed = 2;
// Indicates that it is the device address of node has init arg and do not need device address.
constexpr size_t kDeviceAddressFlagIgnoreDevicePtr = 4;
// Indicates that it is the ptr of device address is nullptr.
constexpr size_t kDeviceAddressFlagNullptr = 8;

class RUNTIME_HARDWARE_EXPORT DeviceAddress : public mindspore::DeviceSync {
 public:
  using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
  DeviceAddress();
  DeviceAddress(void *device_ptr, size_t size);

  explicit DeviceAddress(void *ptr, size_t size, const std::string &device_name);
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id, const std::string &device_name);
  explicit DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId type_id,
                         const std::string &device_name, uint32_t stream_id);
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index, const std::string &device_name);
  explicit DeviceAddress(const DeviceAddress &other);
  DeviceAddress &operator=(const DeviceAddress &) = delete;
  ~DeviceAddress();

  std::string ToString() const;

  DeviceAddressPtr CloneDeviceAddress();

  const void *GetPtr() const;
  void set_ptr(void *ptr);
  size_t GetSize() const override;
  void SetSize(size_t size);

  std::string format() const;
  void set_format(const std::string &format);
  const std::string &padding_type() const;
  void set_padding_type(const std::string &padding_type);
  TypeId type_id() const override;
  void set_type_id(TypeId dtype_id);
  bool from_mem_pool() const;
  void set_from_mem_pool(bool from_mem_pool) const;
  virtual void set_communication_ptr(uint8_t *communication_ptr);
  bool from_persistent_mem() const;
  void set_from_persistent_mem(bool from_persistent_mem);
  bool need_recycle() const;
  void set_need_recycle(bool need_recycle);
  void *GetMutablePtr() const override;
  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const;
  void SetShapeVector(const ShapeVector &shape_vector);

  TensorStorageInfoPtr GetTensorStorageInfo() const override;
  void set_tensor_storage_info(const TensorStorageInfoPtr &tensor_storage_info);

  device::DeviceType GetDeviceType() const override;
  void SetDeviceType(const device::DeviceType &device_type);

  uint32_t device_id() const;

  void set_stream_id(uint32_t stream_id);
  const uint32_t stream_id() const override;

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node);
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const;
  void ClearHeldByNodes();

  void SetNodeIndex(const AnfNodePtr &node, size_t out_index);
  KernelWithIndex GetNodeIndex() const;

  // Return whether DeviceAddress has a valid ptr.
  bool IsPtrValid() const;

  void Swap(DeviceAddress *other);

  // Free the ptr in user data when the ref count is 0.
  void ClearUserData() {}

  std::pair<AnfNodeWeakPtr, size_t> node_index() const;
  void SetDevicePointerDeleter(std::function<void(void *, bool)> &&deleter) override;

  const DevicePointerPtr &device_pointer() const;
  void set_device_pointer(const DevicePointerPtr &ptr_ref_cnt);

  size_t size() const { return size_; }

  void set_allocator(const std::shared_ptr<AddressAllocator> &allocator) { device_pointer_->set_allocator(allocator); }

  std::shared_ptr<AddressAllocator> allocator() const { return device_pointer_->allocator(); }

  void set_data(tensor::TensorDataPtr &&data) override;
  const tensor::TensorDataPtr &data() const override;
  bool has_data() const override;

  void ClearDeviceMemory() override;

 protected:
  // Set a device pointer destructor to kernel tensor, used to release resource reclaiming of the device pointer
  // automatically when DeviceAddress destructed.
  void SetDevicePtrDeleter();

  void *GetDevicePtr() const { return device_pointer_->ptr(); }
  void SetDevicePtr(void *ptr) const { device_pointer_->set_ptr(ptr); }

  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  // The DeviceAddress is held by ValueNodes. These ValueNodes are outputs of forward network.
  // We need to release the device memory when the reference count of the device address in bprop graph is 0.
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes_;

  bool from_persistent_mem_{false};
  bool need_recycle_{false};

  // The padding type corresponds to data format.
  std::string padding_type_;

  // the data for numpy object.
  tensor::TensorDataPtr data_;

  DevicePointerPtr device_pointer_;
  TensorStorageInfoPtr tensor_storage_info_{nullptr};
  uint32_t stream_id_{0};
  size_t size_{0};
  Format format_{Format::DEFAULT_FORMAT};
  // The data enum type id of the KernelTensor.
  TypeId dtype_id_{kTypeUnknown};
  // The device target name, such as "GPU","Ascend".
  device::DeviceType device_type_{device::DeviceType::kUnknown};
  // The origin flatten shape vector for Tensor/Scalar/Tuple/List.
  // 1. For Tensor type, means its shape. For example, a Tensor with shape (8, 16), shape_vector_ is {8, 16}.
  // 2. For Scalar type, shape_vector_ is an empty ShapeVector, i.e. {}.
  // 3. For Tuple/List (all elements must be Tensor with same shape or Scalar) type, the shape_vector_
  // consists of the element number and the shape of element in Tuple/List. For example, if a Tuple of the structure
  // ((8,16), (8,16)) contains two Tensors of shape (8, 16), then shape_vector_ is {2, 8, 16}, 2 means elements
  // number in Tuple/List. A Tuple with a structure such as ((), ()) that contains two Scalar, the shape_vector_ of
  // this Tuple is {2}.
  ShapeVector shape_vector_{};

  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUMemoryManager;
  friend class mindspore::device::cpu::CPUDeviceContext;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::gpu::GPUDeviceContext;
  friend class mindspore::device::gpu::GPUResManager;
  friend class mindspore::device::ascend::AscendRuntimeCore;
  friend class mindspore::device::ascend::AscendMemoryManager;
  friend class mindspore::device::ascend::AscendResManager;
  friend class mindspore::device::ascend::DataDumper;
  friend class mindspore::SingleOpInferSession;
  friend class mindspore::RuntimeUtils;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;

using DevicePtrDeleterMakerFunc = std::function<void(void *, bool)>;
MS_CORE_API void SetDevicePtrDeleterMaker(device::DeviceType device_type, DevicePtrDeleterMakerFunc &&func);

template <device::DeviceType t>
struct DevicePtrDeleterMakerRegister {
  explicit DevicePtrDeleterMakerRegister(DevicePtrDeleterMakerFunc &&maker) {
    SetDevicePtrDeleterMaker(t, std::move(maker));
  }
};

#define REGISTER_DEVICE_PTR_DELETER_MAKER(t, f)                        \
  namespace {                                                          \
  static DevicePtrDeleterMakerRegister<t> g_deleter_maker_register(f); \
  }
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
