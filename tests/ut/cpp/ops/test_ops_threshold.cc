#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, kValueAny}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)}})
    .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, kValueAny}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat64});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt32});
  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(Threshold, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
