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
namespace mindspore::ops {

std::unordered_map<std::string, OpDefPtr> &GetOpDefTable() {
  static std::unordered_map<std::string, OpDefPtr> gOpDefTable;
  return gOpDefTable;
}

OpDefPtr GetOpDef(const std::string &op_name) {
  auto &gOpDefTable = GetOpDefTable();
  auto it = gOpDefTable.find(op_name);
  if (it != gOpDefTable.end()) {
    return it->second;
  }
  return nullptr;
}

void AddOpDef(const std::string &op_name, const OpDefPtr op_def) { (void)GetOpDefTable().emplace(op_name, op_def); }

bool IsPrimitiveFunction(const std::string &op_name) { return GetOpDef(op_name) != nullptr; }

std::vector<OP_DTYPE> GetSourceDtypeByArgHandler(const std::string &arg_handler_func) {
  static std::map<std::string, std::vector<OP_DTYPE>> arg_handler_map = {
    {"to_pair", {OP_DTYPE::DT_INT, OP_DTYPE::DT_FLOAT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_kernel_size", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_strides", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_rates", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_dilations", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_output_padding", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_2d_paddings", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"dtype_to_type_id", {OP_DTYPE::DT_TYPE}},
    {"str_to_enum", {OP_DTYPE::DT_STR}},
  };
  auto iter = arg_handler_map.find(arg_handler_func);
  if (iter == arg_handler_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Miss definition of arg_handler '" << arg_handler_func << "' here.";
  }
  return iter->second;
}
}  // namespace mindspore::ops
