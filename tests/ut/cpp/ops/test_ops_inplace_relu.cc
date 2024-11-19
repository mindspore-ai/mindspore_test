#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat32});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat64}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat64});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeInt32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InplaceReLU, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
