/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/command.h"
#include "minddata/dataset/util/log_adapter.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <stdlib.h>

namespace mindspore {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
const int32_t output_stack_len = 4096;
const int32_t buf_len = 1024;
const int32_t cmd_len = 1024;

std::mutex mtx;

void ExecuteCMD(const std::string &input_cmd) {
  // check the input
  if (input_cmd.size() >= cmd_len) {
    MS_LOG(WARNING) << "ExecuteCMD the input cmd is too long.";
    return;
  }

  // gen the output file
  std::stringstream ss;
  ss << std::this_thread::get_id();
  std::string output_filename = "/tmp/" + std::to_string(getpid()) + "_" + ss.str();

  // combine the whold cmd
  std::string whole_cmd = input_cmd + " >" + output_filename + " 2>&1;";

  // copy the input to local variable
  char cmd[cmd_len] = {0};
  if (strcpy_s(cmd, cmd_len, whole_cmd.c_str()) != EOK) {
    MS_LOG(WARNING) << "ExecuteCMD strcpy_s failed.";
    return;
  }

  {
    // execute the cmd
    std::unique_lock<std::mutex> lock(mtx);
    if (system(cmd) == -1) {
      MS_LOG(WARNING) << "ExecuteCMD system(\"" << cmd << "\", \"r\") failed.";
      return;
    }
  }

  // read the output file
  std::ifstream ifs(output_filename, std::ios::in);
  if (!ifs.is_open()) {
    MS_LOG(WARNING) << "ExecuteCMD read file: " << output_filename << " failed.";
    return;
  }

  int32_t offset = 0;
  char output_stack[output_stack_len] = {0};
  char c = ifs.get();
  while (ifs.good()) {
    output_stack[offset] = c;
    if (offset >= output_stack_len - 2) {
      break;
    }
    offset += 1;
    c = ifs.get();
  }
  ifs.close();

  std::string output_stack_str(output_stack);
  if (output_stack_str.find("command not found") != std::string::npos ||
      output_stack_str.find("No such file or directory") != std::string::npos) {
    MS_LOG(WARNING) << "Please `pip install py-spy` to get the stacks of the stuck process.";
  } else {
    MS_LOG(WARNING) << "Map worker subprocess stack:\n" << output_stack_str;
  }

  // remove the output file
  if (remove(output_filename.c_str()) != 0) {
    MS_LOG(WARNING) << "ExecuteCMD remove file: " << output_filename << " failed.";
  }
}
#endif
}  // namespace dataset
}  // namespace mindspore
