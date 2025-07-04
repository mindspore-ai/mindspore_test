/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_FILE_SYSTEM_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_FILE_SYSTEM_H_

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include "utils/system/base.h"
#include "utils/log_adapter.h"
#include "utils/os.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace system {
class WriteFile;
class PosixWriteFile;
using WriteFilePtr = std::shared_ptr<WriteFile>;
using PosixWriteFilePtr = std::shared_ptr<PosixWriteFile>;
constexpr size_t kMaxFileRWLength = static_cast<size_t>(2047) * 1024 * 1024;
// File system of create or delete directory
class FileSystem {
 public:
  FileSystem() = default;

  virtual ~FileSystem() = default;

  // Create a new read/write file with mode
  virtual WriteFilePtr CreateWriteFile(const string &file_name, const char *mode = "w+") = 0;

  // Check the file is exist?
  virtual bool FileExist(const string &file_name) = 0;

  // Delete the file
  virtual bool DeleteFile(const string &file_name) = 0;

  // Create a directory
  virtual bool CreateDir(const string &dir_name) = 0;

  // Delete the specified directory
  virtual bool DeleteDir(const string &dir_name) = 0;
};

// A file that can be read and write
class WriteFile {
 public:
  explicit WriteFile(const string &file_name) : file_name_(file_name) {}

  virtual ~WriteFile() = default;

  // Open the file using a special mode
  virtual bool Open(const char *mode = "w+") = 0;

  // append the content to file
  virtual bool Write(const std::string &data) {
    MS_LOG(WARNING) << "Attention: Maybe not call the function.";
    return true;
  }

  // Write to a file at a given offset like linux function pwrite
  virtual bool PWrite(const void *buf, size_t nbytes, size_t offset) = 0;

  // Read from a file at a given offset like linux function pwrite
  virtual bool PRead(void *buf, size_t nbytes, size_t offset) = 0;

  // Trunc file size to length
  virtual bool Trunc(size_t length) = 0;

  // Get size of this file
  virtual size_t Size() = 0;

  // name: return the file name
  string get_file_name() { return file_name_; }

  // flush: flush local buffer data to filesystem.
  virtual bool Flush() = 0;

  // sync: sync the content to disk
  virtual bool Sync() = 0;

  // close the file
  virtual bool Close() = 0;

 protected:
  string file_name_;

  // The size of this file.
  size_t size_{0};
};

#if defined(SYSTEM_ENV_POSIX)
// File system of create or delete directory for posix system
class MS_CORE_API PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() = default;

  ~PosixFileSystem() override = default;

  // create a new write file using a special mode
  WriteFilePtr CreateWriteFile(const string &file_name, const char *mode) override;

  // check the file is exist?
  bool FileExist(const string &file_name) override;

  // delete the file
  bool DeleteFile(const string &file_name) override;

  // Create a Directory
  bool CreateDir(const string &dir_name) override;

  // Delete the specified directory.
  bool DeleteDir(const string &dir_name) override;
};

// A file that can be read and write for posix
class PosixWriteFile : public WriteFile {
 public:
  explicit PosixWriteFile(const string &file_name) : WriteFile(file_name), file_(nullptr) {}
  PosixWriteFile(const PosixWriteFile &);
  PosixWriteFile &operator=(const PosixWriteFile &);

  ~PosixWriteFile() override {
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

  bool Open(const char *mode) override;
  bool Write(const std::string &data) override;
  bool PWrite(const void *buf, size_t nbytes, size_t offset) override;
  bool PRead(void *buf, size_t nbytes, size_t offset) override;
  bool Trunc(size_t length) override;
  size_t Size() override;
  bool Close() override;
  bool Flush() override;
  bool Sync() override;

 private:
  bool POperate(const void *write_buf, void *read_buf, size_t nbytes, size_t offset, bool read);

  FILE *file_;
};
#endif

#if defined(SYSTEM_ENV_WINDOWS)
// File system of create or delete directory for windows system
class MS_CORE_API WinFileSystem : public FileSystem {
 public:
  WinFileSystem() = default;

  ~WinFileSystem() override = default;

  // create a new write file with mode
  WriteFilePtr CreateWriteFile(const string &file_name, const char *mode) override;

  // check the file is exist?
  bool FileExist(const string &file_name) override;

  // delete the file
  bool DeleteFile(const string &file_name) override;

  // Create a Directory
  bool CreateDir(const string &dir_name) override;

  // Delete the specified directory.
  bool DeleteDir(const string &dir_name) override;
};

// A file that can be read and write for windows
class WinWriteFile : public WriteFile {
 public:
  explicit WinWriteFile(const string &file_name) : WriteFile(file_name), file_(nullptr) {}

  ~WinWriteFile() override;

  bool Open(const char *mode) override;

  bool Write(const std::string &data) override;

  bool PWrite(const void *buf, size_t nbytes, size_t offset) override;

  bool PRead(void *buf, size_t nbytes, size_t offset) override;

  bool Trunc(size_t length) override;

  size_t Size() override;

  bool Close() override;

  bool Flush() override;

  bool Sync() override;

 private:
  FILE *file_;
};
#endif
}  // namespace system
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYSTEM_FILE_SYSTEM_H_
