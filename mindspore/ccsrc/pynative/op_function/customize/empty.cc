/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <iterator>
#include <memory>
#include <string>
#include "ir/scalar.h"
#include "pynative/op_function/converter.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "pynative/forward/forward_task.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/op_executor.h"
#include "include/common/utils/tensor_utils.h"
#include "pynative/pynative_utils.h"
#include "op_def/auto_generate/gen_ops_def.h"
#include "pynative/op_function/customize/direct_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"

namespace mindspore::pynative {
py::object Pyboost_Empty_OP(const PrimitivePtr &prim, const std::vector<mindspore::ops::OP_DTYPE> &source_type,
                            const ValueTuplePtr &shape, const std::optional<Int64ImmPtr> &dtype,
                            const std::optional<StringImmPtr> &device) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, "Empty", false,
                                     true);
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, 1>();
  auto promises = tensor::TransformPromise(py_output);
  MS_LOG(DEBUG) << "start Empty";
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [shape, dtype, device, promises]() {
      std::string device_name;
      if (device.has_value()) {
        device_name = device.value()->value();
        if (device_name != "CPU" && device_name != "Ascend") {
          MS_LOG(EXCEPTION) << "Only support ['CPU', 'Ascend'] for device, but get '" << device_name << "'";
        }
        MS_LOG(DEBUG) << "Using input device_name: " << device_name;
      } else {
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
        MS_LOG(DEBUG) << "Using default device_name: " << device_name;
      }

      auto device_ctx = runtime::OpRunner::GetDeviceContext(device_name);
      MS_EXCEPTION_IF_NULL(device_ctx);

      TypeId real_type = kNumberTypeFloat32;  // default dtype
      if (dtype.has_value()) {
        real_type = static_cast<TypeId>(dtype.value()->value());
        MS_LOG(DEBUG) << "dtype.has_value == True, input type: " << TypeIdToString(real_type);
      }

      ShapeVector output_shape;
      for (size_t i = 0; i < shape->size(); i++) {
        int64_t shape_i = std::static_pointer_cast<Int64Imm>((*shape)[i])->value();
        output_shape.push_back(shape_i);
      }

      std::vector<tensor::BaseTensorPtr> outputs;
      kernel::pyboost::PyBoostUtils::CreateOutputTensor(real_type, output_shape, &outputs);
      kernel::pyboost::PyBoostUtils::PrepareOpOutputs(device_ctx, 0, outputs);
      tensor::SetPromise(promises, outputs[0]);

      auto fn = [device_ctx, outputs]() { kernel::pyboost::PyBoostUtils::MallocOpOutputs(device_ctx, outputs); };

      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughNoWaitDeviceTask>(fn));
      } else {
        fn();
      }
    },
    [promises]() { tensor::SetException(promises); }));
  MS_LOG(DEBUG) << "finish Empty";

  return py::reinterpret_steal<py::object>(tensor::TransformOutput(py_output));
}

py::object Empty(const py::list &args) {
  static pynative::Converter converter(&ops::gEmpty);
  auto shape = converter.ToIntList<py::tuple>(args, kIndex0);
  auto dtype = converter.ToDtypeOptional(args, kIndex1);
  auto device = converter.ToStringOptional(args, kIndex2);
  return Pyboost_Empty_OP(mindspore::prim::kPrimEmpty, converter.source_type(), shape, dtype, device);
}

py::object Pyboost_Empty_Base(const PrimitivePtr &prim, const py::list &args) {
  return mindspore::pynative::Empty(args);
}
}  // namespace mindspore::pynative
