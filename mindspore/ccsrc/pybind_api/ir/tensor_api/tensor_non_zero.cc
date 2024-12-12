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

#include "pybind_api/ir/arg_handler.h"
#include "pybind_api/ir/tensor_api/tensor_api.h"
#include "pipeline/pynative/op_function/converter.h"

namespace mindspore {
namespace tensor {

py::object TensorMethodNonZero(const py::object &self, const py::args &py_args, const py::kwargs &py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({"NonZero()", "NonZero(bool as_tuple=False)"}, "nonzero");
  py::list arg_list;
  auto sig = parser.Parse(py_args, py_kwargs, &arg_list, true);
  arg_list.insert(0, self);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#ifndef ENABLE_TEST
  switch (sig.index_) {
    case 0:
      MS_LOG(INFO) << "Call TensorNonZero";
      return ToPython(TensorPyboostMethodRegister::GetOp(tensor::TensorPyboostMethod::kNonZeroReg)(arg_list));
      break;
    case 1:
      if (py::cast<bool>(arg_list[kIndex1]) && (backend == kAscendDevice || backend == kDavinciDevice)) {
        MS_LOG(INFO) << "Call TensorNonZeroExt";
        // as_tuple is not required
        arg_list.attr("pop")();
        return ToPython(TensorPyboostMethodRegister::GetOp(tensor::TensorPyboostMethod::kNonZeroExtReg)(arg_list));
      } else if (!py::cast<bool>(arg_list[kIndex1])) {
        MS_LOG(INFO) << "Call TensorNonZero";
        // as_tuple is not required
        arg_list.attr("pop")();
        return ToPython(TensorPyboostMethodRegister::GetOp(tensor::TensorPyboostMethod::kNonZeroReg)(arg_list));
      } else {
        MS_LOG(ERROR) << "Device target is not supported!";
        return py::none();
      }
      break;
    default:
      return py::none();
  }
  return py::none();
#endif
}
}  // namespace tensor
}  // namespace mindspore
