/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include "common/common_test.h"
#include "infer/ops_func_impl/log_softmax_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_INFER_TEST_DECLARE(LogSoftmaxExt, EltwiseOpParams);

OP_FUNC_IMPL_INFER_TEST_CASES(
  LogSoftmaxExt,
  testing::Values(
    EltwiseOpParams{
      {2, 3}, kFloat16, {2, 3}, kFloat16, {CreateScalar<int64_t>(-1), CreateScalar<int64_t>(kNumberTypeFloat16)}},
    EltwiseOpParams{
      {2, -1}, kFloat32, {2, -1}, kFloat32, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat32)}},
    EltwiseOpParams{
      {-1, -1}, kFloat64, {-1, -1}, kFloat64, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat64)}},
    EltwiseOpParams{
      {-2}, kFloat16, {-2}, kFloat32, {CreateScalar<int64_t>(2), CreateScalar<int64_t>(kNumberTypeFloat32)}},
    EltwiseOpParams{{-2}, kInt32, {-2}, kFloat32, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat32)}},
    EltwiseOpParams{
      {2, 3}, kInt8, {2, 3}, kFloat16, {CreateScalar<int64_t>(-1), CreateScalar<int64_t>(kNumberTypeFloat16)}},
    EltwiseOpParams{
      {2, -1}, kInt64, {2, -1}, kFloat32, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat32)}},
    EltwiseOpParams{
      {-1, -1}, kUInt8, {-1, -1}, kFloat64, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat64)}},
    EltwiseOpParams{
      {-2}, kFloat32, {-2}, kFloat32, {CreateScalar<int64_t>(2), CreateScalar<int64_t>(kNumberTypeFloat32)}},
    EltwiseOpParams{
      {-2}, kFloat64, {-2}, kFloat32, {CreateScalar(kValueAny), CreateScalar<int64_t>(kNumberTypeFloat32)}}));
}  // namespace ops
}  // namespace mindspore
