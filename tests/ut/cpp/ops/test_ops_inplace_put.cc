#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 8}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, -1, 8}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 8, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, -1, 8, -1}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 8, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, -1, 8, -1}}, {kNumberTypeBFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 8, -1, 2}, kNumberTypeBool},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 2, 2}, kNumberTypeBool},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{2, -1, 8, -1, 2}}, {kNumberTypeBool});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InplacePut, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
