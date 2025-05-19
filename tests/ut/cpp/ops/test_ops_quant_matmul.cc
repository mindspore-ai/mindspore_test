#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{1, 4, 5}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{2, 3, 5}}, {kNumberTypeInt8});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{4, 5}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt32))},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{2, 3, 5}}, {kNumberTypeInt32});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{1, 4, 5}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt8});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt8});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{1,}, kNumberTypeInt64},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt8});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{1,}, kNumberTypeInt64},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{2, 3, -1}}, {kNumberTypeInt8});

    generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{1, 4, 5}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{1,}, kNumberTypeInt64},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{3,}, kNumberTypeFloat32},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1, -1, 5}}, {kNumberTypeInt8});
  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(QuantMatmul, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
