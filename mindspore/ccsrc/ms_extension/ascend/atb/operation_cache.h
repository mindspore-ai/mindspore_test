// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file includes code sourced from the project "op-plugin".
// Original repository: https://gitee.com/ascend/op-plugin.

#ifndef MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ATB_OPERATION_CACHE_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ATB_OPERATION_CACHE_H_

#include <string.h>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>

#include "atb/atb_infer.h"
#include "ms_extension/pynative/pyboost_extension.h"

#define CHECK_ATB_RET(op, st, func)                                                                       \
  do {                                                                                                    \
    if (st != 0) {                                                                                        \
      MS_LOG(EXCEPTION) << "ATB function [" #func "] result error. st=" << st << ", op is " << op         \
                        << ". See atb logs in \"~/atb/log\" for more details, refer to atb documents at " \
                           "https://www.hiascend.com/document";                                           \
    }                                                                                                     \
  } while (0);

namespace atb {
constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
EXTENSION_EXPORT char *g_hash_buf_ptr();
EXTENSION_EXPORT int &g_hash_offset_ref();

#define MEMCPY_TO_BUF(data_expression, size_expression)                             \
  if (g_hash_offset_ref() + (size_expression) > g_hash_buf_size) {                  \
    g_hash_offset_ref() = g_hash_buf_max_size;                                      \
    return;                                                                         \
  }                                                                                 \
  memcpy(g_hash_buf_ptr() + g_hash_offset_ref(), data_expression, size_expression); \
  g_hash_offset_ref() += size_expression;

EXTENSION_EXPORT uint64_t calc_hash_id();

template <typename T>
void add_param_to_buf(const T &value) {
  MEMCPY_TO_BUF(&value, sizeof(T));
}

EXTENSION_EXPORT void add_param_to_buf(const std::string &s);

// api
template <typename T>
void add_param_to_buf(const std::string &name, const T &value) {
  add_param_to_buf(name);
  add_param_to_buf(value);
}

// api
// Each operator implements its own hash function calculation.
// It is possible to hash only the attributes that may change in the parameters of the calculation.
// following example::
//
// `template <>`
// `struct HashOpParam<atb::infer::XXXParam> {   //if XXXParam's transposeA and hasBias need hash`
//     `void operator()(const atb::infer::XXXParam& param) const {`
//         `add_param_to_buf("transposeA", param.transposeA);`
//         `add_param_to_buf("hasBias", param.hasBias);`
//     `}`
// `};`
template <typename T>
struct HashOpParam {
  void operator()(const T &param) const {
    MS_EXCEPTION(mindspore::NotImplementedError)
      << "The atb::HashOpParam<" << ms::inner::GetFunctionName(typeid(T).name()) << "> is not defined.";
  }
};

template <typename T>
uint64_t computeHash(const T &obj) {
  g_hash_offset_ref() = 0;
  HashOpParam<T>{}(obj);
  return calc_hash_id();
}

// api
template <typename ParamType>
class OpParamCache {
 public:
  static OpParamCache &getInstance() {
    static OpParamCache instance;
    return instance;
  }

  atb::Operation *getOperation(const ParamType &param, const std::string &name) {
    uint64_t hashValue = computeHash(param);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto opCache = op_map_.find(hashValue);
      if (opCache != op_map_.end()) {
        return opCache->second;
      }
      atb::Operation *op = nullptr;
      auto st = atb::CreateOperation(param, &op);
      CHECK_ATB_RET(name, st, CreateOperation);
      MS_EXCEPTION_IF_NULL(op);
      op_map_[hashValue] = op;
      return op;
    }
  }

 private:
  OpParamCache() = default;
  OpParamCache(const OpParamCache &) = delete;
  OpParamCache &operator=(const OpParamCache &) = delete;
  ~OpParamCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &opItem : op_map_) {
      DestroyOperation(opItem.second);
    }
  }
  std::unordered_map<uint64_t, atb::Operation *> op_map_;
  mutable std::mutex mutex_;
};
}  // namespace atb

#endif  // MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ATB_OPERATION_CACHE_H_
