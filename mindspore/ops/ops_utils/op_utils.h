/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OP_UTILS_H
#define MINDSPORE_CORE_OPS_OP_UTILS_H
#include <algorithm>
#include <climits>
#include <memory>
#include <utility>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "include/api/visible.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/value_utils.h"
#include "utils/core_op_utils.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "mindspore/ops/op_def/array_op_name.h"
#include "mindspore/ops/op_def/math_op_name.h"

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#define MS_LIKELY(x) (x)
#else
#define MS_LIKELY(x) __builtin_expect(!!(x), 1)
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif
#define MS_CHECK_VALUE(cond, msg)        \
  do {                                   \
    if (MS_UNLIKELY(!(cond))) {          \
      MS_EXCEPTION(ValueError) << (msg); \
    }                                    \
  } while (0)

namespace mindspore::ops {
constexpr auto kBitSize = 64;
const std::set<TypePtr> common_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                              kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBFloat16};

const std::set<TypePtr> common_valid_types_with_bool = {
  kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool, kBFloat16};

const std::set<TypePtr> common_valid_types_with_complex = {kInt8,    kInt16,     kInt32,      kInt64,   kUInt8,
                                                           kUInt16,  kUInt32,    kUInt64,     kFloat16, kFloat32,
                                                           kFloat64, kComplex64, kComplex128, kBFloat16};

const std::set<TypePtr> common_valid_types_with_complex_and_bool = {
  kInt8,    kInt16,   kInt32,   kInt64,     kUInt8,      kUInt16, kUInt32,  kUInt64,
  kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool,   kBFloat16};

const std::set<TypePtr> common_integral_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
const std::set<TypePtr> common_float_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
const std::set<TypePtr> all_types = {kBool,    kInt,     kInt8,    kInt16,     kInt32,      kInt64,
                                     kUInt,    kUInt8,   kUInt16,  kUInt32,    kUInt64,     kFloat,
                                     kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};
std::vector<int64_t> CalBroadCastShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                       const std::string &op_name, const std::string &op_x_name = "input1",
                                       const std::string &op_y_name = "input2");
OPS_API std::vector<int64_t> CalBroadCastShapeV2(const std::vector<int64_t> &x_shape,
                                                 const std::vector<int64_t> &y_shape, const std::string &op_name,
                                                 const std::string &op_x_name = "input1",
                                                 const std::string &op_y_name = "input2");
abstract::ShapePtr BroadCastInferShape(const std::string &op_name,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);
bool IsBroadcastable(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape);
ShapeVector BroadCastInferShape(const std::string &op_name, const ValuePtrList &input_values);
BaseShapePtr EltwiseGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtr EltwiseGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtrList EltwiseGradSimpleInferType(const PrimitivePtr &primitive, const ValuePtrList &input_values);
ShapeArray EltwiseGradSimpleInferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values);
void ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, std::vector<int64_t> *axis, const size_t dim);
bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive);
ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, bool keep_dims = false);
ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                        const std::vector<int64_t> &axis, bool keep_dims_value = false);
abstract::ShapePtr ReduceBaseInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args,
                                        const std::string &prim_name);
TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args,
                            const std::set<TypePtr> &check_list);
abstract::ShapePtr ReduceExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtr ReduceExtInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args);

BaseShapePtr SetPadShape(const ShapeVector &x_shape, const ArrayValue<int64_t> &paddings);
BaseShapePtr PadInferShapeBase(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                               const size_t pad_dim);

bool ObscureShapeEqual(const ShapeVector &lhs, const ShapeVector &rhs);

// Get the shape value from abstract input arg
// Ops like DynamicBroadcastTo or Reshape can directly get the shape value
// from input which represents shape by invoking this function
// Do not support input with type of AbstractTuple of AbstractTensor
ShapeVector GetShapeValue(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg);

inline ShapeVector ConvertBaseShapeToTensorShape(const BaseShapePtr &base) {
  auto shape_ptr = base->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr->shape();
}

inline ShapeVector GetShapeFromTensor(const AbstractBasePtr &abs) {
  auto base_shape = abs->GetShape();
  return ConvertBaseShapeToTensorShape(base_shape);
}

void CheckSparseShape(ShapeVector sparse_shp, ShapeVector dense_shp);

void CheckSparseShape(const size_t shape_size, const size_t expected_dim, const std::string &arg_name);

void CheckSparseIndicesDtype(const TypePtr data_type, const std::string &arg_name);

void CheckSparseIndicesDtypeInt32(const TypePtr data_type, const std::string &arg_name);

inline void CheckInputShapeEmpty(const std::string &prim_name, const std::vector<AbstractBasePtr> &input_args) {
  for (size_t i = 0; i < input_args.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]->GetShape());
    if (input_args[i]->GetShape()->IsDimZero()) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', input " << i << "'s shape should not be empty!";
    }
  }
}

ShapeVector ConvertToShapeVector(const abstract::AbstractTuplePtr &shape);

template <typename T>
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

template <typename T>
AbstractBasePtr InferSequenceSetItem(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list);

TypePtr HighPriorityType(const TypePtr &x_type, const TypePtr &y_type, const std::string &op_name);

constexpr auto kCSRAvgRows = "csr_avg_rows";
constexpr auto kIsCSR = "is_csr";
constexpr auto kCSRDenseShape = "dense_shape";
constexpr auto kCSRAxis = "axis";
constexpr auto kHasDynamicValue = "has_dynamic_value";

inline int64_t get_batch_rank(const PrimitivePtr &prim) {
  if (prim->HasAttr(kBatchRank)) {
    auto value_ptr = prim->GetAttr(kBatchRank);
    return GetValue<int64_t>(value_ptr);
  }
  return 0;
}

inline int64_t PadModeStringToInt(const std::string &pad) {
  std::string pad_mode = pad;
  (void)std::transform(pad_mode.begin(), pad_mode.end(), pad_mode.begin(), toupper);
  if (pad_mode == "VALID") {
    return static_cast<int64_t>(2);
  } else if (pad_mode == "SAME") {
    return static_cast<int64_t>(1);
  } else if (pad_mode == "PAD") {
    return static_cast<int64_t>(0);
  } else if (pad_mode == "CALCULATED") {
    return static_cast<int64_t>(0);
  } else {
    MS_LOG(EXCEPTION) << "Got an invalid pad_mode string: " << pad_mode << ".";
  }
}

static inline TypePtr PromoteType(TypePtr a, TypePtr b, const std::string &op_name) {
  const auto f32 = kNumberTypeFloat32;
  const auto f16 = kNumberTypeFloat16;
  const auto f64 = kNumberTypeFloat64;
  const auto bf16 = kNumberTypeBFloat16;
  const auto s8 = kNumberTypeInt8;
  const auto u8 = kNumberTypeUInt8;
  const auto s16 = kNumberTypeInt16;
  const auto u16 = kNumberTypeUInt16;
  const auto s32 = kNumberTypeInt32;
  const auto u32 = kNumberTypeUInt32;
  const auto s64 = kNumberTypeInt64;
  const auto u64 = kNumberTypeUInt64;
  const auto b1 = kNumberTypeBool;
  const auto c64 = kNumberTypeComplex64;
  const auto c128 = kNumberTypeComplex128;
  const auto ud = kTypeUnknown;

  static std::unordered_map<TypeId, size_t> typeid_idx = {{f32, 0},  {f16, 1},  {f64, 2}, {bf16, 3}, {s8, 4},
                                                          {u8, 5},   {s16, 6},  {u16, 7}, {s32, 8},  {u32, 9},
                                                          {s64, 10}, {u64, 11}, {b1, 12}, {c64, 13}, {c128, 14}};
  static std::unordered_map<TypeId, TypePtr> typeid_typeptr = {
    {f32, kFloat32}, {f16, kFloat16}, {f64, kFloat64}, {bf16, kBFloat16}, {s8, kInt8},
    {u8, kUInt8},    {s16, kInt16},   {u16, kUInt16},  {s32, kInt32},     {u32, kUInt32},
    {s64, kInt64},   {u64, kUInt64},  {b1, kBool},     {c64, kComplex64}, {c128, kComplex128}};

  TypeId a_type_id;
  if (a->isa<TensorType>()) {
    auto a_tensor_type = a->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(a_tensor_type);
    auto a_element = a_tensor_type->element();
    MS_EXCEPTION_IF_NULL(a_element);
    a_type_id = a_element->type_id();
  } else {
    a_type_id = a->type_id();
  }

  TypeId b_type_id;
  if (b->isa<TensorType>()) {
    auto b_tensor_type = b->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(b_tensor_type);
    auto b_element = b_tensor_type->element();
    MS_EXCEPTION_IF_NULL(b_element);
    b_type_id = b_element->type_id();
  } else {
    b_type_id = b->type_id();
  }

  if (typeid_idx.find(a_type_id) == typeid_idx.end()) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the type " << a->ToString() << "is invalid";
  }

  if (typeid_idx.find(b_type_id) == typeid_idx.end()) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the type " << b->ToString() << "is invalid";
  }

  if (a_type_id == b_type_id) {
    return typeid_typeptr[a_type_id];
  }

  static const std::vector<std::vector<TypeId>> promote_types_lookup = {
    /*         f32  f16  f64  bf16  s8  u8  s16  u16  s32  u32  s64  u64  b1 c64  c128 */
    /* f32 */ {f32, f32, f64, f32, f32, f32, f32, ud, f32, ud, f32, ud, f32, c64, c128},
    /* f16 */ {f32, f16, f64, f32, f16, f16, f16, ud, f16, ud, f16, ud, f16, c64, c128},
    /* f64 */ {f64, f64, f64, f64, f64, f64, f64, ud, f64, ud, f64, ud, f64, c128, c128},
    /* bf16*/ {f32, f32, f64, bf16, bf16, bf16, bf16, ud, bf16, ud, bf16, ud, bf16, c64, c128},
    /* s8  */ {f32, f16, f64, bf16, s8, s16, s16, ud, s32, ud, s64, ud, s8, c64, c128},
    /* u8  */ {f32, f16, f64, bf16, s16, u8, s16, ud, s32, ud, s64, ud, u8, c64, c128},
    /* s16 */ {f32, f16, f64, bf16, s16, s16, s16, ud, s32, ud, s64, ud, s16, c64, c128},
    /* u16 */ {ud, ud, ud, ud, ud, ud, ud, u16, ud, ud, ud, ud, ud, ud, ud},
    /* s32 */ {f32, f16, f64, bf16, s32, s32, s32, ud, s32, ud, s64, ud, s32, c64, c128},
    /* u32 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, u32, ud, ud, ud, ud, ud},
    /* s64 */ {f32, f16, f64, bf16, s64, s64, s64, ud, s64, ud, s64, ud, s64, c64, c128},
    /* u64 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, u64, ud, ud, ud},
    /* b1  */ {f32, f16, f64, bf16, s8, u8, s16, ud, s32, ud, s64, ud, b1, c64, c128},
    /* c64 */ {c64, c64, c128, c64, c64, c64, c64, ud, c64, ud, c64, ud, c64, c64, c128},
    /* c128*/ {c128, c128, c128, c128, c128, c128, c128, ud, c128, ud, c128, ud, c128, c128, c128},
  };

  auto return_type_id = promote_types_lookup[typeid_idx[a_type_id]][typeid_idx[b_type_id]];

  if (return_type_id == ud) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the promote output type is invalid";
  }

  return typeid_typeptr[return_type_id];
}

// map used for pass to identify and replace the op by aclnnview
static const std::map<std::string, std::string> op_enabled_aclnn = {
  {kTransposeOpName, kTransposeViewOpName}, {kSplitOpName, kSplitViewOpName}, {kConcatOpName, kConcatViewOpName}};
// map used for aclnn kernel select, because aclnnview op is not register by yaml.
static const std::map<std::string, std::string> aclnn_view_to_op = {
  {kTransposeViewOpName, kTransposeOpName}, {kSplitViewOpName, kSplitOpName}, {kConcatViewOpName, kConcatOpName}};

void CheckTensorScalarRank(const PrimitivePtr &primitive, const AbstractBasePtr input_arg, const std::string &arg_name);
bool IsFloatType(TypePtr type);
bool IsIntegralType(TypePtr type, bool include_bool);
OPS_API std::vector<int64_t> CalBroadCastShapeV3(const std::vector<int64_t> &x_shape,
                                                 const std::vector<int64_t> &y_shape);
OPS_API int ConvertReductionForAclnn(Reduction reduction);
OPS_API size_t CalOutputSize(const std::vector<int64_t> &output_shape, const size_t &type_size);
OPS_API ValueTuplePtr ConvertShapeVectorToValueTuple(const ShapeVector &shape_vector);
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_UTILS_H
