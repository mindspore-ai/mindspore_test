/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/sig_handler.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/wait.h>
#include <sys/msg.h>
#include <sys/shm.h>
#endif

#include <csignal>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/util/task_manager.h"

namespace mindspore::dataset {
static std::unordered_map<int64_t, std::vector<int>> worker_groups = {};

// The shm_id & msg_id will be registered when the worker process run
// key: the process id
// value: the shm_id / msg_id
// Scenario 1: When the main process is killed, the worker process will release shm & msg in SIGTERMHandler
// Scenario 2: When the worker process is killed, the main process will release shm & msg in SIGCHLDHandler
std::mutex shm_mgs_id_mtx_;                      // lock for g_shm_id & g_msg_id
static std::map<std::string, int32_t> g_shm_id;  // used by map / batch multiprocess mode data transfer
static std::map<std::string, int32_t> g_msg_id;  // used by map / batch multiprocess mode data transfer

#if !defined(_WIN32) && !defined(_WIN64)
/// \brief Set handler for the specified signal.
/// \param[in] signal The signal to set handler.
/// \param[in] handler The handler to execute when receive the signal.
/// \param[in] old_action The former handler.
void SetSignalHandler(int signal, void (*handler)(int, siginfo_t *, void *), struct sigaction *old_action) {
  struct sigaction action {};
  action.sa_sigaction = handler;
  action.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&action.sa_mask) != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to initialise the signal set, " << strerror(errno);
  }
  if (sigaction(signal, &action, old_action) != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to set handler for " << strsignal(signal) << ", " << strerror(errno);
  }
}

/// \brief A signal handler for SIGINT to interrupt watchdog.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGINTHandler(int signal, siginfo_t *info, void *context) {
  // Wake up the watchdog which is designed as async-signal-safe.
  TaskManager::WakeUpWatchDog();
}

/// \brief Release the shared memory and message queue when got signal TERM / CHLD
void ReleaseShmAndMsg() {
  std::string current_pid = std::to_string(getpid());

  {
    std::lock_guard<std::mutex> lock(shm_mgs_id_mtx_);
    // release the shm & msg used by the current process when the main process is killed
    for (auto &item : g_shm_id) {
      // so just release the shm used by the current process
      // scenario 1: for the thread in main process, the item.first is "MainProcessPID_WorkerPID"
      // scenario 2: for the worker process, the item.first is "WorkerPID"
      if (item.first.find(current_pid) == std::string::npos) {
        continue;
      }
      if (item.second != -1) {
        if (shmctl(item.second, IPC_RMID, NULL) == -1 && errno != EINVAL) {
          MS_LOG(ERROR) << "shmctl shm_id: " << std::to_string(item.second)
                        << " error. Errno: " << std::to_string(errno);
        } else {
          MS_LOG(INFO) << "Delete shared memory with shm_id: " << std::to_string(item.second) << " successfully.";
        }
        g_shm_id[item.first] = -1;
      }
    }

    for (auto &item : g_msg_id) {
      // so just release the msg used by the current process
      // scenario 1: for the thread in main process, the item.first is "MainProcessPID_WorkerPID"
      // scenario 2: for the worker process, the item.first is "WorkerPID"
      if (item.first.find(current_pid) == std::string::npos) {
        continue;
      }
      if (item.second != -1) {
        if (msgctl(item.second, IPC_RMID, 0) == -1 && errno != EINVAL) {
          MS_LOG(ERROR) << "Delete msg queue id: " << std::to_string(item.second) << " failed.";
        } else {
          MS_LOG(INFO) << "Delete msg queue id: " << std::to_string(item.second) << " successfully.";
        }
        g_msg_id[item.first] = -1;
      }
    }
  }
}

/// \brief A signal handler for SIGTERM to exit the process.
/// \details When Python exits, it may terminate the children processes before deleting our runtime.
///   Then the watch dog has not been aborted, it will report an error and terminate the main process.
///   So we suppress SIGTERM sent from main process here by _exit(EXIT_SUCCESS).
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGTERMHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGTERM) {
    MS_LOG(ERROR) << "SIGTERMHandler expects SIGTERM signal, but got: " << strsignal(signal);
    _exit(EXIT_FAILURE);
  }

  MS_LOG(INFO) << "Got SIGTERM signal from process: " << info->si_pid;

  // release the shm & msg when the main process is killed
  ReleaseShmAndMsg();

  if (info->si_pid == getppid()) {
    MS_LOG(INFO) << "Dataset worker process " << std::to_string(getpid())
                 << " was terminated by parent process: " << std::to_string(info->si_pid)
                 << ", exits with successful status.";
    _exit(EXIT_SUCCESS);
  }
  // reset the handler to the default
  struct sigaction term_action {};
  term_action.sa_handler = SIG_DFL;
  term_action.sa_flags = 0;
  if (sigemptyset(&term_action.sa_mask) != 0) {
    MS_LOG(ERROR) << "Failed to initialise the signal set, " << strerror(errno);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &term_action, nullptr) != 0) {
    MS_LOG(ERROR) << "Failed to set handler for " << strsignal(signal) << ", " << strerror(errno);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

/// \brief A signal handler for SIGBUS to retrieve the kill information.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGBUSHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGBUS) {
    MS_LOG(ERROR) << "SIGBUSHandler expects SIGBUS signal, but got: " << strsignal(signal);
    _exit(EXIT_FAILURE);
  }

  MS_LOG(INFO) << "Got SIGBUS signal from process: " << info->si_pid;

  if (info->si_code == BUS_ADRERR) {
    MS_LOG(ERROR) << "Unexpected bus error encountered in process: " << std::to_string(getpid())
                  << ". Non-existent physical address. This might be caused by insufficient shared memory. "
                  << "Please check if '/dev/shm' has enough available space via 'df -h'.";
  }

  // reset the handler to the default
  struct sigaction bus_action {};
  bus_action.sa_handler = SIG_DFL;
  bus_action.sa_flags = 0;
  if (sigemptyset(&bus_action.sa_mask) != 0) {
    MS_LOG(ERROR) << "Failed to initialise the signal set, " << strerror(errno);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &bus_action, nullptr) != 0) {
    MS_LOG(ERROR) << "Failed to set handler for " << strsignal(signal) << ", " << strerror(errno);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

/// \brief A signal handler for SIGCHLD to clean the rest processes.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGCHLDHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGCHLD) {
    MS_LOG(ERROR) << "SIGCHLDHandler expects SIGCHLD signal, but got: " << strsignal(signal);
    _exit(EXIT_FAILURE);
  }

  MS_LOG(INFO) << "Got SIGCHLD signal from process: " << info->si_pid;

  for (auto &worker_group : worker_groups) {
    auto &pids = worker_group.second;
    int ppid = pids[0];
    if (getpid() != ppid) {
      continue;  // this worker group is not the children of current process
    }
    for (auto i = 1; i < pids.size(); ++i) {
      int pid = pids[i];
      siginfo_t sig_info{};
      sig_info.si_pid = 0;
      auto error = waitid(P_PID, pid, &sig_info, WEXITED | WNOHANG | WNOWAIT);
      std::string msg;
      if (error < 0) {
        if (errno == ECHILD) {
          msg = "Dataset worker process " + std::to_string(pid) +
                " has already exited. Its state may have been retrieved by other threads.";
        } else {
          MS_LOG(WARNING) << "Failed to wait for dataset worker process " << pid << ", got: " << strerror(errno);
          continue;
        }
      } else {
        if (sig_info.si_pid == 0) {
          continue;  // There were no children in a wait state.
        }
        if (sig_info.si_code == CLD_EXITED && sig_info.si_status != EXIT_SUCCESS) {  // exited unexpected
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) + " exited unexpected with exit code " +
                std::to_string(sig_info.si_status) + ".";
        } else if (sig_info.si_code == CLD_KILLED) {  // killed by signal
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) +
                " was killed by signal: " + std::string(strsignal(sig_info.si_status)) + ".";
        } else if (sig_info.si_code == CLD_DUMPED) {  // core dumped
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) +
                " core dumped: " + std::string(strsignal(sig_info.si_status)) + ".";
        } else {
          MS_LOG(INFO) << "Ignore dataset worker process " << pid << " with signal code " << sig_info.si_code;
          continue;
        }
      }
      auto pids_to_kill = pids;
      pids.clear();  // Clear the monitoring status of the process group before performing a termination.
      for (const auto &pid_to_kill : pids_to_kill) {
        if (pid_to_kill != ppid && pid_to_kill != pid) {
          MS_LOG(INFO) << "Terminating child process: " << pid_to_kill;
          kill(pid_to_kill, SIGTERM);
        }
      }

      // release the shm & msg when the worker process is killed
      ReleaseShmAndMsg();

      MS_LOG(ERROR) << msg << " Main process will be terminated.";
      kill(getpid(), SIGTERM);
      // In case the signal is not responded, return here
      MS_LOG(WARNING) << "Main process may not respond to the SIGTERM signal, please check if it is blocked.";
      return;
    }
  }
  MS_LOG(INFO) << "End got SIGCHLD signal";
}
#endif

void RegisterHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGINT, &SIGINTHandler, nullptr);
  SetSignalHandler(SIGTERM, &SIGINTHandler, nullptr);
#endif
}

void RegisterMainHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGINT, &SIGINTHandler, nullptr);
  SetSignalHandler(SIGBUS, &SIGBUSHandler, nullptr);
  SetSignalHandler(SIGCHLD, &SIGCHLDHandler, nullptr);
#endif
}

void RegisterWorkerHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGBUS, &SIGBUSHandler, nullptr);
  SetSignalHandler(SIGTERM, &SIGTERMHandler, nullptr);
#endif
}

std::string GetPIDsString(const std::vector<int> &pids) {
  std::string pids_string = "[";
  for (auto itr = pids.begin(); itr != pids.end(); ++itr) {
    if (itr != pids.begin()) {
      pids_string += ", ";
    }
    pids_string += std::to_string(*itr);
  }
  pids_string += "]";
  return pids_string;
}

void RegisterWorkerPIDs(int64_t id, const std::vector<int> &pids) {
  MS_LOG(INFO) << "Watch dog starts monitoring process(es): " << GetPIDsString(pids);
  worker_groups[id] = pids;
}

void DeregisterWorkerPIDs(int64_t id) {
  MS_LOG(INFO) << "Watch dog stops monitoring process(es): " << GetPIDsString(worker_groups[id]);
  (void)worker_groups.erase(id);
}

void RegisterShmIDAndMsgID(std::string pid, int32_t shm_id, int32_t msg_id) {
  {
    std::lock_guard<std::mutex> lock(shm_mgs_id_mtx_);
    g_shm_id[pid] = shm_id;
    g_msg_id[pid] = msg_id;
  }
  MS_LOG(INFO) << "Update the shm_id to " << std::to_string(shm_id) << ", msg_id to " << std::to_string(msg_id)
               << " for pid: " << pid;
}
}  // namespace mindspore::dataset
