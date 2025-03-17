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

#include "tests/ut/cpp/operator/meta_dsl/api_test/api_define.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"

namespace mindspore::prim {
/** Python code:
 *  def check_isinstance(x):
 *    return isinstance(x, Tensor), isinstance(x, (int, float))
 */
BeginFunction(TestIsInstance, x) {
  auto res1 = IsInstance(x, TypeId::kObjectTypeTensorType);
  auto res2 = IsInstance(x, {TypeId::kNumberTypeInt, TypeId::kNumberTypeFloat});
  Return(Tuple(res1, res2));
}
EndFunction(TestIsInstance)

/** Python code:
 *  def if_simple(x):
 *    if x > 0:
 *      return 1
 *    return 0
 */
BeginFunction(TestIf, x) {
  auto true_branch = [&]() { Return(Value(1)); };
  auto false_branch = [&]() { Return(Value(0)); };
  Return(If(Call(Prim(Greater), x, Value(0)), true_branch, false_branch, ()));
}
EndFunction(TestIf)

/** Python code:
 *  def if_exp(x, y):
 *    if x is not None:
 *      return x + y
 *    return None
 */
BeginFunction(TestIfExp, x, y) {
  auto true_branch = [&]() { Return(Call(Prim(Add), x, y)); };
  auto false_branch = [&]() { Return(Value(kNone)); };
  Return(If(IsNotNone(x), true_branch, false_branch, (x, y)));
}
EndFunction(TestIfExp)

/** Python code:
 *  def custom_bprop(x, y, out, dout):
 *    return zeros_like(x), zeros_like(y)
 */
BeginFunction(TestCustomBprop, x, y, out, dout) {
  auto dx = Call(Prim(ZerosLike), x);
  auto dy = Call(Prim(ZerosLike), y);
  Return(Tuple(dx, dy));
}
EndFunction(TestCustomBprop)

/** Python code:
 *  def for_func(x, lower, upper):
 *    def cumsum(index, res):
 *      return index + res
 *
 *    for i in range(lower, upper):
 *      x = cumsum(i, x)
 *    return x
 */
BeginFunction(TestFor, x, lower, upper) {
  auto cumsum = [&](const NodePtr &index, const NodePtr &res) { Return(Call(Prim(Add), index, res)); };
  auto out = For(lower, upper, cumsum, x);
  Return(out);
}
EndFunction(TestFor)

/** Python code:
 *  def while_func(x):
 *    while x < 100:
 *      x = x + 1
 *    return x
 */
BeginFunction(TestWhile, x) {
  auto cond_func = [&](const NodePtr &x) { Return(Less(x, Value(100))); };
  auto loop_func = [&](const NodePtr &x) { Return(Call(Prim(Add), x, Value(1))); };
  auto out = While(cond_func, loop_func, x);
  Return(out);
}
EndFunction(TestWhile)

/** Python code:
 *  def scan_func(init, xs):
 *    def cumsum(res, elem):
 *      res = res + elem
 *      return res, res
 *
 *    res = init
 *    ys = []
 *    for x in xs:
 *      res, y = cumsum(res, x)
 *      ys.append(y)
 *    return res, ys
 */
BeginFunction(TestScan, init, xs) {
  auto cumsum = [&](const NodePtr &x, const NodePtr &elem) {
    auto res = Call(Prim(Add), x, elem);
    Return(Tuple(res, res));
  };
  auto out = Scan(cumsum, init, xs);
  Return(out);
}
EndFunction(TestScan)

/** Python code:
 *  def func(x, y):
 *    return x and y
 */
BeginFunction(TestAnd, x, y) {
  auto out = And(x, y);
  Return(out);
}
EndFunction(TestAnd)

/** Python code:
 *  def func(x, y):
 *    return x or y
 */
BeginFunction(TestOr, x, y) {
  auto out = Or(x, y);
  Return(out);
}
EndFunction(TestOr)
}  // namespace mindspore::prim
