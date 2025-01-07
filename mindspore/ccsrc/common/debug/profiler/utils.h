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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <fcntl.h>
#include <libgen.h>
#include <linux/limits.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <string>

#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define LIKELY(x) __builtin_expect(!!(x), 1)

namespace mindspore {
namespace profiler {
namespace ascend {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
class Utils {
 public:
  static bool IsFileExist(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
  }

  static bool IsFileWritable(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), W_OK) == 0) ? true : false;
  }

  static bool IsDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    struct stat st = {0};
    int ret = lstat(path.c_str(), &st);
    if (ret != 0) {
      return false;
    }
    return S_ISDIR(st.st_mode) ? true : false;
  }

  static bool CreateDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    if (IsFileExist(path)) {
      return IsDir(path) ? true : false;
    }
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
      std::string base_dir = path.substr(0, ++pos);
      if (IsFileExist(base_dir)) {
        if (IsDir(base_dir)) {
          continue;
        } else {
          return false;
        }
      }
      if (mkdir(base_dir.c_str(), 0750) != 0) {
        return false;
      }
    }
    return (mkdir(path.c_str(), 0750) == 0) ? true : false;
  }

  static std::string RealPath(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
      return "";
    }
    return std::string(realPath);
  }

  static std::string RelativeToAbsPath(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return "";
    }
    if (path[0] != '/') {
      char pwd_path[PATH_MAX] = {0};
      if (getcwd(pwd_path, PATH_MAX) != nullptr) {
        return std::string(pwd_path) + "/" + path;
      }
      return "";
    }
    return std::string(path);
  }

  static std::string DirName(const std::string &path) {
    if (path.empty()) {
      return "";
    }
    std::string temp_path = std::string(path.begin(), path.end());
    char *path_c = dirname(const_cast<char *>(temp_path.data()));
    return path_c ? std::string(path_c) : "";
  }

  static uint64_t GetClockMonotonicRawNs() {
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 +
           static_cast<uint64_t>(ts.tv_nsec);  // 1000000000为秒转换为纳秒的倍数
  }

  static uint64_t getClockSyscnt() {
    uint64_t cycles;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#elif defined(__x86_64__)
    constexpr uint32_t uint32Bits = 32U;
    uint32_t hi = 0;
    uint32_t lo = 0;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#elif defined(__arm__)
    const uint32_t uint32Bits = 32U;
    uint32_t hi = 0;
    uint32_t lo = 0;
    asm volatile("mrrc p15, 1, %0, %1, c14" : "=r"(lo), "=r"(hi));
    cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#else
    cycles = 0;
#endif
    return cycles;
  }

  static bool CreateFile(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX || !CreateDir(DirName(path))) {
      return false;
    }
    int fd = creat(path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP);
    return (fd < 0 || close(fd) != 0) ? false : true;
  }
};
#else
class Utils {
 public:
  static bool IsFileExist(const std::string &path) { return true; }
  static bool IsFileWritable(const std::string &path) { return true; }
  static bool IsDir(const std::string &path) { return true; }
  static bool CreateDir(const std::string &path) { return true; }
  static std::string RealPath(const std::string &path) { return ""; }
  static std::string RelativeToAbsPath(const std::string &path) { return ""; }
  static std::string DirName(const std::string &path) { return ""; }
  static uint64_t GetClockMonotonicRawNs() { return 0; }
  static uint64_t getClockSyscnt() { return 0; }
  static bool CreateFile(const std::string &path) { return true; }
};
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
