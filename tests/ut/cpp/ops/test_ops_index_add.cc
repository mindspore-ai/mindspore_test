#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{0}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat16, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{0}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{0}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, 3, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{0}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, 3, -1, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{0, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3, -1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat16});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(IndexAddExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
