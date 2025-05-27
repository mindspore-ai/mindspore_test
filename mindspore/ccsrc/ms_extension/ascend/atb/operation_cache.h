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
#include <map>
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
  } while (0)

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
  g_hash_offset_ref() += size_expression

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

class EXTENSION_EXPORT OpParamCacheBase {
 public:
  virtual ~OpParamCacheBase() = default;
};

class EXTENSION_EXPORT AtbContextManager {
  template <typename ParamType>
  class OpParamCache;

 public:
  static AtbContextManager &GetInstance() {
    static AtbContextManager ins;
    return ins;
  }

  atb::Context *GetContext(void *stream) {
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    auto &ctx = ctx_map_[stream];
    if (ctx == nullptr) {
      auto st = atb::CreateContext(&ctx);
      CHECK_ATB_RET("", st, CreateContext);
      st = ctx->SetExecuteStream(static_cast<aclrtStream>(stream));
      CHECK_ATB_RET("", st, SetExecuteStream);
    }
    return ctx;
  }

  template <typename ParamType>
  atb::Operation *GetOperation(const ParamType &param, const std::string &name) {
    auto cache = GetOperationCache(param);
    MS_EXCEPTION_IF_NULL(cache);
    return cache->getOperation(param, name);
  }

  ~AtbContextManager() {
    // all operations must be freed before context
    op_param_caches_.clear();
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    for (auto &iter : ctx_map_) {
      auto st = atb::DestroyContext(iter.second);
      CHECK_ATB_RET("", st, DestroyContext);
    }
  }
  AtbContextManager(const AtbContextManager &) = delete;
  AtbContextManager &operator=(const AtbContextManager &) = delete;

 private:
  template <typename ParamType>
  OpParamCache<ParamType> *GetOperationCache(const ParamType &param) {
    std::lock_guard<std::mutex> lock(op_mutex_);
    auto type_idx = std::type_index(typeid(ParamType));
    if (op_param_caches_.find(type_idx) == op_param_caches_.end()) {
      op_param_caches_[type_idx] = std::make_unique<OpParamCache<ParamType>>();
    }
    return static_cast<OpParamCache<ParamType> *>(op_param_caches_[type_idx].get());
  }
  AtbContextManager() = default;
  std::map<void *, atb::Context *> ctx_map_;
  std::map<std::type_index, std::unique_ptr<OpParamCacheBase>> op_param_caches_;
  mutable std::mutex ctx_mutex_;
  mutable std::mutex op_mutex_;
};

template <typename ParamType>
class AtbContextManager::OpParamCache : public OpParamCacheBase {
 public:
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

  ~OpParamCache() override {
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
