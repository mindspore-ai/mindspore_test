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

#include "ops/op_def.h"
#include <iostream>
#include <memory>
#include "mindspore/ops/op_def/other_op_name.h"
#include "utils/ms_context.h"

namespace mindspore::ops {

std::unordered_map<std::string, OpDefPtr> &GetOpDefTable() {
  static std::unordered_map<std::string, OpDefPtr> gOpDefTable;
  return gOpDefTable;
}

OpDefPtr GetCustomOpDef(const std::string &op_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // The input type of the custom operator is defined as a tuple type to support PyBoost of the custom operator;
  // Graph mode does not use this definition.
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode ||
      ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    return nullptr;
  }

  auto &gOpDefTable = GetOpDefTable();
  auto it = gOpDefTable.find(op_name);
  if (it != gOpDefTable.end()) {
    return it->second;
  }
  return nullptr;
}

OpDefPtr GetOpDef(const std::string &op_name) {
  if (op_name == kCustomOpName) {
    return GetCustomOpDef(op_name);
  }
  auto &gOpDefTable = GetOpDefTable();
  auto it = gOpDefTable.find(op_name);
  if (it != gOpDefTable.end()) {
    return it->second;
  }
  return nullptr;
}

void AddOpDef(const std::string &op_name, const OpDefPtr op_def) { (void)GetOpDefTable().emplace(op_name, op_def); }

bool IsPrimitiveFunction(const std::string &op_name) { return GetOpDef(op_name) != nullptr; }
}  // namespace mindspore::ops
