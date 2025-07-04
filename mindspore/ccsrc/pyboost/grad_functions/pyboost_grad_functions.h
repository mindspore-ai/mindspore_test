/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PYBOOST_GRAD_FUNCTIONS_H
#define MINDSPORE_PYBOOST_GRAD_FUNCTIONS_H

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "runtime/pynative/op_runner.h"
#include "mindspore/ccsrc/pyboost/grad_functions/func_object.h"
#include "backend/graph_compiler/op_backend.h"

namespace mindspore::runtime {
using Func = std::function<void(OpRunnerInfo *, VectorRef *)>;
using DoGradFunc = void (*)(OpRunnerInfo *op_runner_info, VectorRef *op_outputs);

void COMMON_EXPORT RegisterDoGradFunc(const DoGradFunc &clone_func);

class PyBoostOpExecute {
 public:
  static PYBOOST_API PyBoostOpExecute &GetInstance();

  // Register pyboost grad op function
  void Register(const std::string &key, Func func) { grad_op_func_map_[key] = func; }

  // Check grad op have already registered
  bool PYBOOST_API IsPyBoostOpRegistered(const std::string &op_name);

  // Unified op run entry for pynative grad
  void PYBOOST_API Execute(OpRunnerInfo *op_runner_info, VectorRef *op_outputs);

  // Api for outside call
  void PYBOOST_API RunPyBoostCall(OpRunnerInfo *op_runner_info, VectorRef *op_outputs);

  // Run op by single op graph
  void PYBOOST_API RunOpDeprecated(OpRunnerInfo *op_runner_info, VectorRef *op_outputs);

  // Clear backend for fork process.
  void ClearBackend() {}

 private:
  // RunOp in VM
  void RunOpInVm(OpRunnerInfo *op_runner_info, VectorRef *op_outputs);

  compile::OpBackend op_backend_;
  std::unordered_map<std::string, FuncObject> grad_op_func_map_;
};

class PyBoostGradOpRegistrar {
 public:
  PyBoostGradOpRegistrar(const std::string &name, const Func &func) {
    PyBoostOpExecute::GetInstance().Register(name, func);
  }
  ~PyBoostGradOpRegistrar() = default;
};

#define MS_REG_PYBOOST_GRAD_OP(NAME, FUNC) static const PyBoostGradOpRegistrar g_##NAME##_pyboost(#NAME, FUNC);
}  // namespace mindspore::runtime
#endif  // MINDSPORE_PYBOOST_GRAD_FUNCTIONS_H
