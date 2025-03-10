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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_DEPRECATED_ENV_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_DEPRECATED_ENV_H_
namespace mindspore {
namespace dump {
void CheckDeprecatedDumpEnv();
void PrintDeprecatedWarning(bool, bool);
}  // namespace dump
}  // namespace mindspore
#endif
