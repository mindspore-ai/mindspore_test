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

#include "tests/ut/cpp/operator/meta_dsl/pixel_shuffle.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore::prim {
void CheckPixelShuffleInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const auto &input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (!IsDynamicRank(input_shape) && input_shape.size() < 3) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects input to have at least 3 dimensions, but got input with "
                             << input_shape.size() << " dimension(s)";
  }
}

/**
 * Python code for comparison:
 * def pixel_shuffle(input, upscale_factor):
 *   if upscale_factor < 0:
 *     raise ValueError(f"For 'PixelShuffle', the 'upscale_factor' must be > 0.")
 *   idx = shape_(input)
 *   length = input.ndim
 *   pre = idx[:-3]
 *   c, h, w = idx[-3:]
 *   if c % upscale_factor ** 2 != 0:
 *     raise ValueError("For 'PixelShuffle', the length of third to last dimension is not divisible"
 *                      "by `upscale_factor` squared.")
 *   c = c // upscale_factor ** 2
 *   input_perm = (pre + (c, upscale_factor, upscale_factor, h, w))
 *   input = ops.Reshape()(input, input_perm)
 *   input_perm = [i for i in range(length - 2)]
 *   input_perm = input_perm + [length, length - 2, length + 1, length - 1]
 *   input_perm = tuple(input_perm)
 *   input = ops.Transpose()(input, input_perm)
 *   input = ops.Reshape()(input, (pre + (c, upscale_factor * h, upscale_factor * w)))
 *   return input
 **/
BeginFunction(PixelShuffle, input, upscale_factor) {
  auto idx = Call(Prim(Shape), input);
  auto length = Call(Prim(Rank), input);
  auto c = GetItem(idx, Call(Prim(ScalarSub), length, Value(3)));
  auto h = GetItem(idx, Call(Prim(ScalarSub), length, Value(2)));
  auto w = GetItem(idx, Call(Prim(ScalarSub), length, Value(1)));

  // Raise error if upscale_factor is not a positive integer.
  auto invalid_upscale_factor = [&]() {
    Return(Raise("ValueError", "For 'PixelShuffle', the 'upscale_factor' must be > 0."));
  };
  // Raise error if dimension is invalid.
  auto invalid_dimension = [&]() {
    Return(Raise("ValueError",
                 "For 'PixelShuffle', the length of third to last dimension is not divisible"
                 "by `upscale_factor` squared."));
  };
  // Operator concatenation.
  auto inner_impl = [&]() {
    auto pre = GetItem(idx, Call(Prim(MakeSlice), Value(0), Call(Prim(ScalarSub), length, Value(3)), Value(1)));
    c = Call(Prim(ScalarFloorDiv), c, Call(Prim(ScalarPow), upscale_factor, Value(2)));
    auto input_perm = Call(Prim(SequenceAdd), pre, Call(Prim(MakeTuple), c, upscale_factor, upscale_factor, h, w));
    input = Call(Prim(Reshape), input, input_perm);
    input_perm = Call(Prim(MakeRange), Call(Prim(ScalarSub), length, Value(2)));
    input_perm = Call(Prim(SequenceAdd), input_perm,
                      Call(Prim(MakeTuple), length, Call(Prim(ScalarSub), length, Value(2)),
                           Call(Prim(ScalarAdd), length, Value(1)), Call(Prim(ScalarSub), length, Value(1))));
    input = Call(Prim(Transpose), input, input_perm);
    auto new_shape = Call(
      Prim(SequenceAdd), pre,
      Call(Prim(MakeTuple), c, Call(Prim(ScalarMul), upscale_factor, h), Call(Prim(ScalarMul), upscale_factor, w)));
    input = Call(Prim(Reshape), input, new_shape);
    Return(input);
  };
  // Implement.
  auto impl_branch = [&]() {
    auto calc_out = Call(Prim(ScalarMod), c, Call(Prim(ScalarPow), upscale_factor, Value(2)));
    auto condition = NotEqual(calc_out, Value(0));
    Return(If(condition, invalid_dimension, inner_impl, (input, upscale_factor, length, c, h, w)));
  };
  Return(If(LessEqual(upscale_factor, Value(0)), invalid_upscale_factor, impl_branch,
            (input, upscale_factor, length, c, h, w)));
}
EndFunction(PixelShuffle)
}  // namespace mindspore::prim
