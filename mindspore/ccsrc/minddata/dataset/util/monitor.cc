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
#include "minddata/dataset/util/monitor.h"

namespace mindspore {
namespace dataset {
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_WIN64)
Status MonitorSubprocess(int pid) {
  CHECK_FAIL_RETURN_UNEXPECTED(pid != -1, "[Internal ERROR] The subprocess id is -1.");
  // blocking execution and get the state changes in a child of the calling process
  MS_LOG(INFO) << "[Monitor] Begin monitor the sub-process: " + std::to_string(pid);
  int status = 0;
  auto ret = waitpid(pid, &status, 0);
  MS_LOG(INFO) << "[Monitor] sub-process id: " << ret << ", errno: " << errno << ", status: " << status;
  if (ret == pid) {  // the status of subprocess is changed
    uint32_t uint_status = static_cast<uint32_t>(status);
    if (WIFEXITED(uint_status)) {  // the child is exit normal
      std::string err_msg = "[Monitor] The sub-process: " + std::to_string(pid) +
                            " exits. Status: " + std::to_string(WEXITSTATUS(uint_status)) +
                            ". Errno: " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else if (WIFSIGNALED(uint_status)) {  // if the child process was terminated by a signal
      std::string err_msg = "[Monitor] The sub-process: " + std::to_string(pid) +
                            " is terminated by a signal abnormally. Signal: " + std::to_string(WTERMSIG(uint_status)) +
                            ". Errno: " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else if (WIFSTOPPED(uint_status)) {  // if the child process was stopped by delivery of a signal
      std::string err_msg =
        "[Monitor] The sub-process: " + std::to_string(pid) +
        " is stopped by delivery of a signal abnormally. Signal: " + std::to_string(WSTOPSIG(uint_status)) +
        ". Errno: " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else if (WIFCONTINUED(uint_status)) {  // returns true if the child process was resumed by delivery of SIGCONT
      MS_LOG(INFO) << "[Monitor] The sub-process: " + std::to_string(pid) + " is resumed.";
    } else {
      MS_LOG(INFO) << "[Monitor] The sub-process: " + std::to_string(pid) +
                        " has generated a new status: " + std::to_string(uint_status);
    }
  } else {
    MS_LOG(INFO) << "[Monitor] The sub-process: " + std::to_string(pid) + " is not exists.";
  }
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
