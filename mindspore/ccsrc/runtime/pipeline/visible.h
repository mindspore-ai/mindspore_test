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

#ifndef MINDSPORE_CCSRC_RUNTIME_PIPELINE_VISIBLE_H_
#define MINDSPORE_CCSRC_RUNTIME_PIPELINE_VISIBLE_H_

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef RUNTIME_PIPELINE_DLL
#define RUNTIME_PIPELINE_EXPORT __declspec(dllexport)
#else
#define RUNTIME_PIPELINE_EXPORT __declspec(dllimport)
#endif
#define RUNTIME_PIPELINE_LOCAL
#else
#define RUNTIME_PIPELINE_EXPORT __attribute__((visibility("default")))
#define RUNTIME_PIPELINE_LOCAL __attribute__((visibility("hidden")))
#endif

#endif  // MINDSPORE_CCSRC_RUNTIME_PIPELINE_VISIBLE_H_
