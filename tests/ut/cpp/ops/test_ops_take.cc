#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{2}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{2}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, -1, 3}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, -1, 3}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 3, 7}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeInt64}})
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Take, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
