/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_
#define MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_

#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <utility>
#include "ms_extension/common/tensor.h"
#include "mindspore/ccsrc/debug/profiler/profiler.h"
#include "mindspore/ccsrc/include/common/utils/tensor_utils.h"
#include "mindspore/ccsrc/pynative/pynative_utils.h"

namespace mindspore {
namespace stub {
class StubNode;
using StubNodePtr = std::shared_ptr<StubNode>;
}  // namespace stub
namespace device {
class DeviceContext;
}
namespace runtime {
class PyBoostDeviceTask;
}
}  // namespace mindspore

namespace ms {
namespace inner {

/**
 * @brief Retrieves the demangled function name (if applicable) from a mangled symbol name.
 * @param name A mangled symbol name.
 * @return The demangled function name (or the original name if demangling is not available).
 */
EXTENSION_EXPORT std::string GetFunctionName(const char *name);

/**
 * @brief Sets a promise for a stub node to associate it with a Tensor output.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param output The Tensor output to associate with the promise.
 */
EXTENSION_EXPORT void SetPromise(const std::string &op_name, const std::tuple<mindspore::stub::StubNodePtr> &tuple,
                                 const ms::Tensor &output);

/**
 * @brief Helper function for setting promises for multiple outputs.
 * @tparam Tuple The type of the tuple containing stub nodes.
 * @tparam Index The indices of the tuple elements.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param outputs The Tensor outputs to associate with the promises.
 */
template <typename Tuple, size_t... Index>
void SetPromiseHelper(const std::string &op_name, const Tuple &tuple, const std::vector<ms::Tensor> &outputs,
                      std::index_sequence<Index...>) {
  (SetPromise(op_name, std::get<Index>(tuple), outputs[Index]), ...);
}

/**
 * @brief Sets promises for multiple outputs based on a tuple of stub nodes.
 * @tparam T The types of the elements in the tuple.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param outputs The Tensor outputs to associate with the promises.
 * @throws If the tuple size does not match the size of the outputs.
 */
template <typename... T>
void SetPromise(const std::string &op_name, const std::tuple<T...> &tuple, const std::vector<ms::Tensor> &outputs) {
  constexpr size_t tuple_size = sizeof...(T);
  if (tuple_size != outputs.size()) {
    MS_LOG(EXCEPTION) << "For op " << op_name << ", the output size should be equal to " << tuple_size << ", but got "
                      << outputs.size();
  }
  SetPromiseHelper(op_name, tuple, outputs, std::make_index_sequence<tuple_size>{});
}

/**
 * @brief Converts a Tensor or an optional Tensor's stub node into a Tensor object.
 * @tparam T The type of the argument, which can be a Tensor or an optional Tensor.
 * @param arg The argument to convert.
 */
template <typename T>
void ConvertMsTensor(const T &arg) {
  if constexpr (std::is_same_v<T, ms::Tensor>) {
    arg.ConvertStubNodeToTensor();
  } else {
    if constexpr (std::is_same_v<T, std::optional<ms::Tensor>>) {
      if (arg.has_value()) {
        arg.value().ConvertStubNodeToTensor();
      }
    }
  }
}

/**
 * @brief Converts multiple stub nodes into Tensor objects.
 * @tparam Args The types of the arguments.
 * @param args The arguments to convert.
 */
template <typename... Args>
void ConvertStubNodeToTensor(const Args &... args) {
  (ConvertMsTensor(args), ...);
}

/**
 * @brief Memory block structure for managing device memory allocation.
 */
struct MemBlock {
  /**
   * @brief Constructs a MemBlock and allocates memory on the device.
   * @param device_context The device context for memory allocation.
   * @param size The size of the memory block to allocate.
   * @param stream_id The stream ID for the memory allocation.
   * @throws If memory allocation fails.
   */
  MemBlock(const mindspore::device::DeviceContext *device_context, size_t size, uint32_t stream_id);

  /**
   * @brief Destructor for MemBlock. Frees the allocated memory.
   */
  ~MemBlock();

  // Pointer to the allocated memory block.
  void *ptr_;
  // The device context used for allocation.
  const mindspore::device::DeviceContext *device_context_;
};
using MemBlockPtr = std::shared_ptr<MemBlock>;
}  // namespace inner

namespace pynative {
/**
 * @class PyboostRunner
 * @brief [API] Represents a runner for PyBoost operations, providing methods to manage execution, memory allocation,
 * and kernel launches.
 */
class EXTENSION_EXPORT PyboostRunner : public std::enable_shared_from_this<PyboostRunner> {
 public:
  /**
   * @brief Constructs a PyboostRunner with the specified operation name.
   * @param op_name The name of the operation.
   */
  explicit PyboostRunner(const std::string &op_name) : _op_name_(op_name) {}

  /**
   * @brief Virtual destructor for PyboostRunner.
   */
  virtual ~PyboostRunner() = default;

  /**
   * @brief [API] Executes a given function and manages its output as PyBoost operation.
   * @tparam OUT_NUM The number of output tensors expected from the function.
   * @tparam FuncType The type of the function to be executed.
   * @tparam Args The types of the arguments to be passed to the function.
   * @param func The function to execute. It should return a result compatible with `SetPromise`.
   * @param args The arguments to pass to the function.
   * @return A Python object representing the outputs of the operation.
   *
   * @details This method:
   * - Records profiling information for the operation.
   * - Converts `StubNode` inputs to `Tensor` objects.
   * - Executes the provided function with the given arguments.
   * - Sets promises for the output tensors.
   * - Handles exceptions via a fallback mechanism.
   *
   * @throws If the function execution or promise setting fails, the exception is propagated.
   */
  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM > 0), py::object>::type Call(FuncType func, Args &&... args) {
    auto op_name = inner::GetFunctionName(typeid(FuncType).name());
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kRunOp, op_name);
    auto py_output = mindspore::tensor::MakeTuple<mindspore::tensor::TensorWrapper, OUT_NUM>();
    auto promises = mindspore::tensor::TransformPromise(py_output);
    mindspore::pynative::DispatchOp(std::make_shared<mindspore::pynative::PassthroughFrontendTask>(
      [=]() {
        inner::ConvertStubNodeToTensor(args...);
        auto result = func(args...);
        inner::SetPromise(op_name, promises, result);
      },
      [promises]() { mindspore::tensor::SetException(promises); }));
    return py::reinterpret_steal<py::object>(mindspore::tensor::TransformOutput(py_output));
  }

  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM == 0), py::object>::type Call(FuncType func, Args &&... args) {
    auto op_name = inner::GetFunctionName(typeid(FuncType).name());
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kRunOp, op_name);
    mindspore::pynative::DispatchOp(std::make_shared<mindspore::pynative::PassthroughFrontendTask>([=]() {
      inner::ConvertStubNodeToTensor(args...);
      func(args...);  // Call the function without return value
    }));
    return py::none();
  }

  /**
   * @brief [API] Executes the PyBoost operation with the given inputs and outputs.
   * @param inputs The input tensors.
   * @param outputs The output tensors.
   */
  void Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);

  /**
   * @brief [API] Calculates the workspace size required for the operation.
   * @return The size of the workspace in bytes. Default is 0.
   */
  virtual size_t CalcWorkspace() { return 0; }

  /**
   * @brief [API] Launches the kernel for the operation.
   */
  virtual void LaunchKernel() = 0;

  /**
   * @brief [API] Retrieves the operation name associated with this runner.
   * @return The operation name as a string.
   */
  const std::string &op_name() const { return _op_name_; }

  /**
   * @brief [API] Retrieves the input tensors.
   * @return A reference to the vector of input tensors.
   */
  const std::vector<ms::Tensor> &inputs() const { return _inputs_; }

  /**
   * @brief [API] Retrieves the output tensors.
   * @return A reference to the vector of output tensors.
   */
  const std::vector<ms::Tensor> &outputs() const { return _outputs_; }

  /**
   * @brief [API] Retrieves the stream ID associated with this runner.
   * @return The stream ID as an unsigned integer.
   */
  uint32_t stream_id() const { return _stream_id_; }

  /**
   * @brief [API] Retrieves the stream associated with this runner.
   * @return A pointer to the stream.
   */
  void *stream() { return _stream_; }

  /**
   * @brief [API] Retrieves the workspace pointer for the operation.
   * @return A pointer to the workspace memory.
   */
  void *workspace_ptr() { return _workspace_ptr_; }

 protected:
  /**
   * @brief Run with pyboost pipeline
   */
  virtual void _Run();

  /**
   * @brief Set stream id and stream ptr of pyboost runner.
   */
  virtual void _PrepareStream();

  /**
   * @brief Prepares device addresses for input and output tensors.
   */
  virtual void _PrepareDeviceAddress();

  /**
   * @brief Allocates tensors memory and workspace memory.
   */
  virtual inner::MemBlockPtr _MallocDeviceAddress();

  /**
   * @brief Dispatch a launch task
   */
  virtual void _DispatchLaunchTask();

  // The name of the operation.
  std::string _op_name_;
  // The input tensors.
  std::vector<ms::Tensor> _inputs_;
  // The output tensors.
  std::vector<ms::Tensor> _outputs_;
  // The stream ID for the operation.
  size_t _stream_id_{0};
  // The stream associated with the operation.
  void *_stream_{nullptr};
  // The device context for the operation.
  mindspore::device::DeviceContext *_device_context_{nullptr};
  // Pointer to the workspace memory.
  void *_workspace_ptr_{nullptr};
};

}  // namespace pynative
}  // namespace ms
#endif  // MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_
