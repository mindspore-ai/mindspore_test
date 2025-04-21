/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_UTILS_MS_UTILS_H_
#define MINDSPORE_CORE_UTILS_MS_UTILS_H_

#include <map>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <limits>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <set>
#include <sstream>
#include "mindapi/base/macros.h"
namespace mindspore {
class MSLogTime {
 public:
  MSLogTime() {}
  ~MSLogTime() {}
  inline void Start() { this->start = std::chrono::system_clock::now(); }
  inline void End() { this->end = std::chrono::system_clock::now(); }
  uint64_t GetRunTimeUS() {
    auto ms_duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end - this->start);
    uint64_t ms = ms_duration.count();
    return ms;
  }

 private:
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
};
}  // namespace mindspore

#define DISABLE_COPY_AND_ASSIGN(ClassType) \
  ClassType(const ClassType &) = delete;   \
  ClassType &operator=(const ClassType &) = delete;

#define TRY_AND_CATCH_WITH_EXCEPTION(expr, error_msg)                               \
  do {                                                                              \
    try {                                                                           \
      (expr);                                                                       \
    } catch (const std::exception &e) {                                             \
      MS_LOG(EXCEPTION) << "Caught exception of " << e.what() << ". " << error_msg; \
    }                                                                               \
  } while (0)

namespace mindspore {
namespace common {
inline const char *SafeCStr(const std::string &str) { return str.c_str(); }
MS_CORE_API const char *SafeCStr(const std::string &&str);

// Memory dev config.
const char kAllocConf[] = "MS_ALLOC_CONF";
const char kAllocAclAllocator[] = "acl_allocator";
const char kAllocSomasWholeBlock[] = "somas_whole_block";
const char kAllocEnableVmm[] = "enable_vmm";
const char kAllocVmmAlignSize[] = "vmm_align_size";
const char kAllocMemoryRecycle[] = "memory_recycle";
const char kAllocMemoryTracker[] = "memory_tracker";
const char kAllocDefragMemoryStepFreq[] = "defrag_memory_step_freq";
const char kAllocMemoryPool[] = "older_pool";

// Runtime dev config.
const char kRuntimeConf[] = "MS_DEV_RUNTIME_CONF";
const char kRuntimeInline[] = "inline";
const char kRuntimeSwitchInline[] = "switch_inline";
const char kRuntimeNewRefCount[] = "new_ref_count";
const char kRuntimeControlFlowOptimize[] = "control_flow_optimize";
const char kRuntimeMultiStream[] = "multi_stream";
const char kRuntimePipeline[] = "pipeline";
const char kRuntimeGraphPipeline[] = "graph_pipeline";
const char kRuntimeKbkSubGraphMode[] = "kbk_sub_graph_mode";
const char kRuntimeCommunicationLaunchGroup[] = "communication_launch_group";
const char kRuntimeView[] = "view";
const char kRuntimeInsertTensorMove[] = "insert_tensormove";
const char kRuntimeAllfinite[] = "all_finite";
const char kRuntimeParalletAssignAddOpt[] = "parallel_assignadd_opt";
const char kRuntimeGeKernel[] = "ge_kernel";
const char kAclnnViewOp[] = "MS_DEV_VIEW_OP";
const char kRuntimeCache[] = "backend_compile_cache";
const char kRuntimeCopyAsync[] = "copy_async";
const char kRuntimeClusterThreadNum[] = "cluster_thread_num";
const char kRuntimeThreadLoadCache[] = "multi_thread_load_cache";
const char kRuntimeAsyncInitComm[] = "async_init_comm";
const char kRuntimeCpuAffinityList[] = "cpu_affinity_list";
const char kRuntimeCpuAffinityMoudule[] = "cpu_affinity_module";
const char kRuntimeActorThreadFixBind[] = "actor_thread_fix_bind";
const char kRuntimeInputOptimize[] = "input_optimize";
const char kRuntimeCommInitLcclOnly[] = "comm_init_lccl_only";
// Runtime debug config.
const char kRuntimeMemoryTrack[] = "memory_track";
const char kRuntimeMemoryStat[] = "memory_statistics";
const char kRuntimeCompileStat[] = "compile_statistics";
const char kRuntimePerformanceStat[] = "performance_statistics";
const char kRuntimePerformanceStatTopNum[] = "performance_statistics_top_num";
const char kRuntimeAclnnCacheQueueLength[] = "aclnn_cache_queue_length";
const char kRuntimeAclnnCache[] = "aclnn_cache";
const char kRuntimePreBuildCommKernel[] = "pre_build_comm_kernel";
const char kRuntimeExecutionOrderCheckIteration[] = "execution_order_check_iteration";
const char kSingleQuote = '\'';
const char kDoubleQuote = '"';
const char kSemicolon = ';';
const char kComma = ',';
const char kColon = ':';
const char kWhiteSpace[] = " \t";
MS_CORE_API void ResetConfig(const std::string &config);
MS_CORE_API std::string GetConfigValue(const std::string &config, const std::string &config_key);
MS_CORE_API bool IsEnableRuntimeConfig(const std::string &runtime_config);
MS_CORE_API bool IsDisableRuntimeConfig(const std::string &runtime_config);
MS_CORE_API std::string GetRuntimeConfigValue(const std::string &runtime_config);
MS_CORE_API std::string GetAllocConfigValue(const std::string &alloc_config);
MS_CORE_API bool IsEnableAllocConfig(const std::string &alloc_config);
MS_CORE_API bool IsDisableAllocConfig(const std::string &alloc_config);
MS_CORE_API bool IsEnableAclnnViewOp(const std::string &op);

// Get env thread safe with cache.
struct EnvHelper;
using EnvHelperPtr = std::shared_ptr<EnvHelper>;
struct MS_CORE_API EnvHelper {
  static EnvHelperPtr &GetInstance();

  const char *GetEnv(const char *conf, bool cache_env = false);

  // Reset env cache, if conf is nullptr, reset all cache.
  void ResetCache(const char *conf);

  std::map<std::string, std::string> env_cache_;
  std::mutex mutex_;
};

static inline std::string GetEnv(const std::string &envvar, const std::string &default_value = "") {
  const char *value = EnvHelper::GetInstance()->GetEnv(envvar.c_str());

  if (value == nullptr) {
    return default_value;
  }

  return std::string(value);
}

static inline int SetEnv(const char *envname, const char *envvar, int overwrite = 1) {
#if defined(_WIN32)
  return 0;
#else
  EnvHelper::GetInstance()->ResetCache(envname);
  return ::setenv(envname, envvar, overwrite);
#endif
}

static inline void SetOMPThreadNum() {
  const size_t kOMPThreadMaxNum = 16;
  const size_t kOMPThreadMinNum = 1;
  // The actor concurrent execution max num.
  const size_t kActorConcurrentMaxNum = 4;

  size_t cpu_core_num = std::thread::hardware_concurrency();
  size_t cpu_core_num_half = cpu_core_num / 2;
  // Ensure that the calculated number of OMP threads is at most half the number of CPU cores.
  size_t OMP_thread_num = cpu_core_num_half / kActorConcurrentMaxNum;

  OMP_thread_num = OMP_thread_num < kOMPThreadMinNum ? kOMPThreadMinNum : OMP_thread_num;
  OMP_thread_num = OMP_thread_num > kOMPThreadMaxNum ? kOMPThreadMaxNum : OMP_thread_num;

  std::string OMP_env = std::to_string(OMP_thread_num);
  (void)SetEnv("OMP_NUM_THREADS", OMP_env.c_str(), 0);
}

static inline bool IsLittleByteOrder() {
  uint32_t check_code = 0x12345678;
  auto check_pointer = reinterpret_cast<uint8_t *>(&check_code);
  uint8_t head_code = 0x78;
  if (check_pointer[0] == head_code) {
    return true;
  }
  return false;
}

static inline bool UseMPI() {
  // If these OpenMPI environment variables are set, we consider this process is launched by OpenMPI.
  std::string ompi_command_env = GetEnv("OMPI_COMMAND");
  std::string pmix_rank_env = GetEnv("PMIX_RANK");
  if (!ompi_command_env.empty() && !pmix_rank_env.empty()) {
    if (!GetEnv("MS_ROLE").empty()) {
      return false;
    }
    return true;
  }
  return false;
}

static inline bool UseDynamicCluster() {
  // If environment variable 'MS_ROLE' or 'MS_SCHED_HOST' is set, we consider this process is participating in cluster
  // building.
  return !common::GetEnv("MS_ROLE").empty() || !common::GetEnv("MS_SCHED_HOST").empty();
}

// UseDynamicCluster or UseMPI. If false, means use rank table file.
static inline bool UseHostCollective() { return common::UseDynamicCluster() || common::UseMPI(); }

template <typename T>
bool IsEqual(const T *a, const T *b) {
  if (a == b) {
    return true;
  }
  if (a == nullptr || b == nullptr) {
    return false;
  }
  return *a == *b;
}

template <typename T>
bool IsEqual(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) {
  return IsEqual(a.get(), b.get());
}

template <typename T>
bool IsAttrsEqual(const T &a, const T &b) {
  if (&a == &b) {
    return true;
  }
  if (a.size() != b.size()) {
    return false;
  }
  auto iter1 = a.begin();
  auto iter2 = b.begin();
  while (iter1 != a.end()) {
    if (iter1->first != iter2->first) {
      return false;
    }
    if (!IsEqual(iter1->second, iter2->second)) {
      return false;
    }
    ++iter1;
    ++iter2;
  }
  return true;
}

inline bool IsFloatEqual(const float &a, const float &b) {
  return (std::fabs(a - b) <= std::numeric_limits<float>::epsilon());
}

inline bool IsDoubleEqual(const double &a, const double &b) {
  return (std::fabs(a - b) <= std::numeric_limits<double>::epsilon());
}

inline bool IsStrNumeric(const std::string &str) {
  return std::all_of(str.begin(), str.end(), [](char c) { return std::isdigit(c); });
}

inline bool IsNeedMemoryStatistic() {
  static const char kMemoryStatistic[] = "MS_MEMORY_STATISTIC";
  static const auto need_statistic = GetEnv(kMemoryStatistic);
  return !need_statistic.empty() && need_statistic != "0";
}

inline bool IsCompileSimulation() {
  static const auto kSimulationLevel = "MS_SIMULATION_LEVEL";
  static const auto kSimulationLevelCompileGraph = "0";
  static const auto kSimulationLevelCompileKernel = "1";
  static const auto kSimulationLevelCompileWithDeivce = "2";
  static const auto simulation_level = common::GetEnv(kSimulationLevel);
  static const auto simu_compile =
    (simulation_level == kSimulationLevelCompileGraph || simulation_level == kSimulationLevelCompileKernel ||
     simulation_level == kSimulationLevelCompileWithDeivce);
  return simu_compile;
}

inline bool IsExecuteSimulation() {
  static const auto kSimulationLevel = "MS_SIMULATION_LEVEL";
  static const auto kSimulationLevelExecute = "3";
  static const auto simulation_level = common::GetEnv(kSimulationLevel);
  static const auto simu_execute = (simulation_level == kSimulationLevelExecute);
  return simu_execute;
}

inline bool IsDryRun() {
  static const char kLaunchSkippedEnv[] = "MS_KERNEL_LAUNCH_SKIP";
  static const auto launch_skipped = GetEnv(kLaunchSkippedEnv);
  static const bool skip_launch = (launch_skipped == "all" || launch_skipped == "ALL" || IsCompileSimulation());
  return skip_launch;
}

inline void SplitString(const std::string &str, char delim, std::set<std::string> *output_list) {
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      output_list->emplace(item);
    }
  }
}
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_UTILS_H_
