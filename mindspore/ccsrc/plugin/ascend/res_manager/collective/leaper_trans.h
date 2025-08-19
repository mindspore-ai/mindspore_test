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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LEAPER_TRANS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LEAPER_TRANS_H_

#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <string>

#include "utils/log_adapter.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {

struct LeaperConnInfo {
  std::vector<int> recv_fds;
  std::vector<int> send_fds;
};

class EXPORT_WRAPPER LeaperTrans {
 public:
  static LeaperTrans &GetInstance();

  LeaperConnInfo Connect(std::string src_ip, std::string dst_ip, uint16_t src_port, uint16_t dst_port);
  bool SendRecv(const void *send_data, void *recv_data, size_t send_size, size_t recv_size,
                const LeaperConnInfo &conn_info);

 private:
  LeaperTrans();
  ~LeaperTrans() = default;

  void SendData(int fd, const uint8_t *data, size_t size);
  void RecvData(int fd, uint8_t *data, size_t size);
  void ClearConnInfo(LeaperConnInfo *conn_info);

  std::mutex send_recv_lock_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LEAPER_TRANS_H_
