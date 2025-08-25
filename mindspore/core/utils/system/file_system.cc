/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "utils/system/file_system.h"
#include "utils/file_utils.h"
#include "utils/convert_utils_base.h"

#if defined(SYSTEM_ENV_POSIX)
#include <sys/stat.h>
#include <unistd.h>
#elif defined(SYSTEM_ENV_WINDOWS)
#include <direct.h>
#endif

namespace {
static const int MAX_DIRECTORY_LENGTH = 1024;
static const int MAX_FILENAME_LENGTH = 128;
static const int MAX_OS_FILENAME_LENGTH = 255;

inline std::string ErrnoToString(const int error_number) {
  std::ostringstream ret_info;
  ret_info << " Errno: " << error_number;
#if defined(__APPLE__)
  char err_info[MAX_FILENAME_LENGTH];
  (void)strerror_r(error_number, err_info, sizeof(err_info));
  ret_info << ", ErrInfo: " << err_info;
#elif defined(SYSTEM_ENV_POSIX)
  char err_info[MAX_FILENAME_LENGTH];
  char *ret = strerror_r(error_number, err_info, sizeof(err_info));
  if (ret != nullptr) {
    ret_info << ", ErrInfo: " << ret;
  }
#elif defined(SYSTEM_ENV_WINDOWS)
  char err_info[MAX_FILENAME_LENGTH];
  (void)strerror_s(err_info, sizeof(err_info), error_number);
  ret_info << ", ErrInfo: " << err_info;
#endif
  return ret_info.str();
}
}  // namespace

namespace mindspore {
namespace system {
#if defined(SYSTEM_ENV_POSIX)
// Implement the Posix file system
WriteFilePtr PosixFileSystem::CreateWriteFile(const string &file_name, const char *mode) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Create write file failed because the file name is null.";
    return nullptr;
  }
  auto fp = std::make_shared<PosixWriteFile>(file_name);
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Create write file(" << file_name << ") failed.";
    return nullptr;
  }
  bool result = fp->Open(mode);
  if (!result) {
    MS_LOG(ERROR) << "Open the write file(" << file_name << ") failed.";
    return nullptr;
  }
  ChangeFileMode(file_name, S_IWUSR | S_IRUSR);
  return fp;
}

bool PosixFileSystem::FileExist(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = access(file_name.c_str(), F_OK);
  if (result != 0) {
    MS_LOG(DEBUG) << "The file(" << file_name << ") not exist.";
    return false;
  }
  return true;
}

bool PosixFileSystem::DeleteFile(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = unlink(file_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the file(" << file_name << ") failed." << ErrnoToString(errno);
    return false;
  }
  return true;
}

static const int DEFAULT_MKDIR_MODE = 0700;
bool PosixFileSystem::CreateDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = mkdir(dir_name.c_str(), DEFAULT_MKDIR_MODE);
  if (result != 0) {
    if (errno != EEXIST) {
      MS_LOG(ERROR) << "Create the dir(" << dir_name << ") failed." << ErrnoToString(errno);
      return false;
    }
  }
  return true;
}

bool PosixFileSystem::DeleteDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = rmdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the dir(" << dir_name << ") failed." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool PosixWriteFile::Open(const char *mode) {
  if (file_ != nullptr) {
    MS_LOG(WARNING) << "The File(" << file_name_ << ") already open.";
    return true;
  }
  // check the path
  if (file_name_.c_str() == nullptr) {
    MS_LOG(EXCEPTION) << "The file path is null.";
  }
  if (file_name_.size() >= PATH_MAX) {
    MS_LOG(EXCEPTION) << "The file name is too long, file name is " << file_name_ << ".";
  }

  // open the file
  file_ = fopen(file_name_.c_str(), mode);
  if (file_ == nullptr) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR. " << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool PosixWriteFile::Write(const std::string &data) {
  MS_LOG(DEBUG) << "Write data(" << data.size() << ") to file(" << this->file_name_ << ").";
  size_t r = fwrite(data.data(), 1, data.size(), file_);
  if (r != data.size()) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR. " << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool PosixWriteFile::PWrite(const void *buf, size_t nbytes, size_t offset) {
  MS_LOG(DEBUG) << "Write data(" << nbytes << ") at offset(" << offset << ")to file(" << file_name_ << ").";
  return POperate(buf, nullptr, nbytes, offset, false);
}

bool PosixWriteFile::PRead(void *buf, size_t nbytes, size_t offset) {
  MS_LOG(DEBUG) << "Read data(" << nbytes << ") at offset(" << offset << ")to file(" << file_name_ << ").";
  return POperate(nullptr, buf, nbytes, offset, true);
}

bool PosixWriteFile::Trunc(size_t length) {
  MS_LOG(DEBUG) << "Trunc file(" << file_name_ << ") to size(" << length << ")";
  if (length == size_) {
    return true;
  }
  if (ftruncate(fileno(file_), length) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") Trunc ERROR. " << ErrnoToString(errno);
    return false;
  }
  size_ = length;
  return true;
}

size_t PosixWriteFile::Size() { return size_; }

bool PosixWriteFile::Close() {
  if (file_ == nullptr) {
    MS_LOG(INFO) << "File(" << file_name_ << ") already close.";
    return true;
  }
  bool result = true;
  if (fclose(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR. " << ErrnoToString(errno);
    result = false;
  }
  file_ = nullptr;
  return result;
}

bool PosixWriteFile::Flush() {
  if (fflush(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR. " << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool PosixWriteFile::Sync() { return Flush(); }

bool PosixWriteFile::POperate(const void *write_buf, void *read_buf, size_t nbytes, size_t offset, bool read) {
  size_t left = nbytes;
  size_t buff_offset = 0;
  auto fd = fileno(file_);
  while (left > 0) {
    size_t length = 0;
    if (left > kMaxFileRWLength) {
      length = kMaxFileRWLength;
    } else {
      length = left;
    }
    left -= length;
    ssize_t r = 0;
    if (read && read_buf != nullptr) {
      auto buff_p = static_cast<uint8_t *>(read_buf) + buff_offset;
      r = pread(fd, buff_p, length, SizeToLong(offset + buff_offset));
    } else if (write_buf != nullptr) {
      auto buff_p = static_cast<const uint8_t *>(write_buf) + buff_offset;
      r = pwrite(fd, buff_p, length, SizeToLong(offset + buff_offset));
    }
    if (r >= 0 && LongToSize(r) != length) {
      MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR. " << ErrnoToString(errno);
      return false;
    }
    buff_offset += length;
  }
  return true;
}

#endif

#if defined(SYSTEM_ENV_WINDOWS)

// Implement the Windows file system

WriteFilePtr WinFileSystem::CreateWriteFile(const string &file_name, const char *mode) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Create write file failed because the file name is null.";
    return nullptr;
  }
  auto fp = std::make_shared<WinWriteFile>(file_name);
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Create write file(" << file_name << ") failed.";
    return nullptr;
  }
  bool result = fp->Open(mode);
  if (!result) {
    MS_LOG(ERROR) << "Open the write file(" << file_name << ") failed.";
    return nullptr;
  }
  ChangeFileMode(file_name, S_IWUSR | S_IRUSR);
  return fp;
}

bool WinFileSystem::FileExist(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = access(file_name.c_str(), F_OK);
  if (result != 0) {
    MS_LOG(DEBUG) << "The file(" << file_name << ") not exist.";
    return false;
  }
  return true;
}

bool WinFileSystem::CreateDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = mkdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Create the dir(" << dir_name << ") is failed, error(" << result << ").";
    return false;
  }
  return true;
}

bool WinFileSystem::DeleteDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = rmdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the dir(" << dir_name << ") is failed, error(" << result << ").";
    return false;
  }
  return true;
}

bool WinFileSystem::DeleteFile(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = unlink(file_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the file(" << file_name << ") is failed." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::Open(const char *mode) {
  if (file_ != nullptr) {
    MS_LOG(WARNING) << "The File(" << file_name_ << ") already open.";
    return true;
  }
  // check the path
  if (file_name_.c_str() == nullptr) {
    MS_LOG(EXCEPTION) << "The file path is null.";
  }
  if (file_name_.size() >= PATH_MAX) {
    MS_LOG(EXCEPTION) << "The file name is too long, file name is " << file_name_ << ".";
  }

  // open the file
  file_ = fopen(file_name_.c_str(), mode);
  if (file_ == nullptr) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::Write(const std::string &data) {
  MS_LOG(DEBUG) << "Write data(" << data.size() << ") to file(" << this->file_name_ << ").";
  size_t r = fwrite(data.data(), 1, data.size(), file_);
  if (r != data.size()) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::PWrite(const void *buf, size_t nbytes, size_t offset) { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::PRead(void *buf, size_t nbytes, size_t offset) { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::Trunc(size_t length) { MS_LOG(EXCEPTION) << "Not Implement"; }

size_t WinWriteFile::Size() { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::Close() {
  if (file_ == nullptr) {
    MS_LOG(WARNING) << "File(" << file_name_ << ") already close.";
    return true;
  }
  bool result = true;
  if (fclose(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    result = false;
  }
  file_ = nullptr;
  return result;
}

bool WinWriteFile::Flush() {
  if (fflush(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR: " << EBADF << ".";
    return false;
  }
  return true;
}

bool WinWriteFile::Sync() { return Flush(); }

WinWriteFile::~WinWriteFile() {
  try {
    if (file_ != nullptr) {
      (void)fclose(file_);
      file_ = nullptr;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when closing file.";
  } catch (...) {
    MS_LOG(ERROR) << "Non standard exception when closing file.";
  }
}
#endif
}  // namespace system
}  // namespace mindspore
