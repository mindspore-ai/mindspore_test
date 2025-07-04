/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_TASK_QUEUE_H
#define ATB_SPEED_TASK_QUEUE_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace atb_speed {

using Task = std::function<void()>;

class TaskQueue {
public:
    void Enqueue(const Task &task);
    Task Dequeue();

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
};
}  // namespace atb_speed

#endif  // ATB_SPEED_TASK_QUEUE_H
