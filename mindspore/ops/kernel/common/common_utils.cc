/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "common/common_utils.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "common/oplib/oplib.h"
#include "common/format_utils.h"
#include "common/debug/common.h"
#include "mindapi/base/type_id.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr char kTypeInt32[] = "Int32";
constexpr auto kQuad = 4;
constexpr size_t kInputFirstIndex = 0;
}  // namespace

size_t GetOutputNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &type = node->Type();
  if (type == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failed to get type in node:" << node->fullname_with_scope();
  } else if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    if (tuple_type->dynamic_len()) {
      return 1;
    }
    const auto &sub_types = tuple_type->elements();
    return static_cast<size_t>(std::count_if(sub_types.begin(), sub_types.end(), [](const TypePtr &sub_type) {
      return sub_type != nullptr && (!sub_type->isa<MonadType>());
    }));
  } else if (type->isa<List>()) {
    auto list_type = type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(list_type);
    if (list_type->dynamic_len()) {
      return 1;
    }
    const auto &sub_types = list_type->elements();
    return static_cast<size_t>(std::count_if(sub_types.begin(), sub_types.end(), [](const TypePtr &sub_type) {
      return sub_type != nullptr && (!sub_type->isa<MonadType>());
    }));
  } else if (type->isa<CSRTensorType>()) {
    return 5;
  } else if (type->isa<COOTensorType>()) {
    return 4;
  }
  return 1;
}

int CalDiagOffset(int diag_index, int max_diag_len, int inner_rows, int inner_cols,
                  const std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> &alignment) {
  bool right_align_super_diagonal = (alignment.first == MatrixDiag::RIGHT);
  bool right_align_sub_diagonal = (alignment.second == MatrixDiag::RIGHT);
  const bool right_align =
    (diag_index >= 0 && right_align_super_diagonal) || (diag_index <= 0 && right_align_sub_diagonal);
  const int diag_len = std::min(inner_rows + std::min(0, diag_index), inner_cols - std::max(0, diag_index));
  const int offset = (right_align) ? (max_diag_len - diag_len) : 0;
  return offset;
}

TypeId DtypeToTypeId(const std::string &dtypes) {
  if (dtypes == "float") {
    return TypeId::kNumberTypeFloat32;
  }
  if (dtypes.empty()) {
    return TypeId::kMetaTypeNone;
  }
  return StringToTypeId(dtypes);
}

std::string Dtype2ShortType(const std::string &dtype) {
  static const std::unordered_map<std::string, std::string> dtype_shortdtype_map = {
    {"float16", "f16"}, {"float32", "f32"}, {"float64", "f64"},  {"int8", "i8"},    {"int16", "i16"},
    {"int32", "i32"},   {"int64", "i64"},   {"uint8", "u8"},     {"uint16", "u16"}, {"uint32", "u32"},
    {"uint64", "u64"},  {"bool", "bool"},   {"bfloat16", "bf16"}};

  auto iter = dtype_shortdtype_map.find(dtype);
  if (iter != dtype_shortdtype_map.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtype;
  }
}

size_t GetDtypeNbyte(const std::string &dtype) {
  static const std::unordered_map<std::string, size_t> dtype_nbyte_map = {
    {"float16", sizeof(float) / 2},   {"float32", sizeof(float)},     {"float64", sizeof(float) * 2},
    {"int8", sizeof(int) / kQuad},    {"int16", sizeof(int) / 2},     {"int32", sizeof(int)},
    {"int64", sizeof(int) * 2},       {"uint8", sizeof(int) / kQuad}, {"uint16", sizeof(int) / 2},
    {"uint32", sizeof(int)},          {"uint64", sizeof(int) * 2},    {"bool", sizeof(char)},
    {"complex64", sizeof(float) * 2}, {"bfloat16", sizeof(float) / 2}};

  auto iter = dtype_nbyte_map.find(dtype);
  if (iter != dtype_nbyte_map.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtype;
  }
}

bool IsSameShape(const ShapeVector &shape_a, const ShapeVector &shape_b) { return shape_a == shape_b; }

bool CheckShapesSame(const ShapeArray &shape_array) {
  auto first_shape = shape_array[0];
  return std::all_of(shape_array.begin() + 1, shape_array.end(),
                     [&first_shape](const ShapeVector &shape) { return IsSameShape(shape, first_shape); });
}

std::vector<TypeId> GetOutputObjectTypeListFromKernelAttr(const kernel::KernelAttr &kernel_attr) {
  size_t output_attr_size = kernel_attr.GetOutputSize();
  std::vector<TypeId> res;
  for (size_t i = 0; i < output_attr_size; ++i) {
    res.push_back(kernel_attr.GetOutputAttr(i).object_type);
  }
  return res;
}

std::vector<TypeId> GetInputObjectTypeListFromKernelAttr(const kernel::KernelAttr &kernel_attr) {
  size_t input_attr_size = kernel_attr.GetInputSize();
  std::vector<TypeId> res;
  for (size_t i = 0; i < input_attr_size; ++i) {
    res.push_back(kernel_attr.GetInputAttr(i).object_type);
  }
  return res;
}

KernelObjectType TypeIdToKernelObjectType(const TypeId &type_id) {
  std::unordered_map<TypeId, KernelObjectType> trans_map{{kObjectTypeTuple, KernelObjectType::TUPLE},
                                                         {kObjectTypeNumber, KernelObjectType::SCALAR},
                                                         {kObjectTypeTensorType, KernelObjectType::TENSOR}};
  if (trans_map.find(type_id) == trans_map.end()) {
    MS_LOG(DEBUG) << "Unsupported type id " << TypeIdToString(type_id)
                  << ", that cannot converted to corresponding kernel object type.";
    return KernelObjectType::UNKNOWN_TYPE;
  }
  return trans_map[type_id];
}

std::vector<KernelObjectType> TypeIdToKernelObjectType(const std::vector<TypeId> &type_ids) {
  std::vector<KernelObjectType> ret;
  (void)std::transform(type_ids.begin(), type_ids.end(), std::back_inserter(ret),
                       [](const TypeId &type_id) { return kernel::TypeIdToKernelObjectType(type_id); });
  return ret;
}

KernelObjectType TypeIdToKernelObjectTypeForTupleUnfold(const TypeId &type_id) {
  std::unordered_map<TypeId, KernelObjectType> trans_map{{kObjectTypeTuple, KernelObjectType::TUPLE_UNFOLD},
                                                         {kObjectTypeNumber, KernelObjectType::SCALAR},
                                                         {kObjectTypeTensorType, KernelObjectType::TENSOR}};
  if (trans_map.find(type_id) == trans_map.end()) {
    MS_LOG(DEBUG) << "Unsupported type id " << TypeIdToString(type_id)
                  << ", that cannot converted to corresponding kernel object type.";
    return KernelObjectType::UNKNOWN_TYPE;
  }
  return trans_map[type_id];
}

std::vector<KernelObjectType> TypeIdToKernelObjectTypeForTupleUnfold(const std::vector<TypeId> &type_ids) {
  std::vector<KernelObjectType> ret;
  (void)std::transform(type_ids.begin(), type_ids.end(), std::back_inserter(ret),
                       [](const TypeId &type_id) { return kernel::TypeIdToKernelObjectTypeForTupleUnfold(type_id); });
  return ret;
}

TypeId KernelObjectTypeToTypeId(const KernelObjectType &object_type) {
  std::unordered_map<KernelObjectType, TypeId> trans_map{{KernelObjectType::TUPLE, kObjectTypeTuple},
                                                         {KernelObjectType::TUPLE_UNFOLD, kObjectTypeTuple},
                                                         {KernelObjectType::SCALAR, kObjectTypeNumber},
                                                         {KernelObjectType::TENSOR, kObjectTypeTensorType}};
  if (trans_map.find(object_type) == trans_map.end()) {
    MS_LOG(DEBUG) << "Unsupported kernel object type " << object_type
                  << ", that cannot converted to corresponding type id.";
    return kTypeUnknown;
  }
  return trans_map[object_type];
}

// The allsame/skip_check and the unequal size scenario don't support object type backoff and use the object_types,
// other scenes support the object type backoff and use the selected_object_types.
std::vector<KernelObjectType> CalKernelObjectTypes(const std::vector<TypeId> &object_types,
                                                   const std::vector<TypeId> &selected_object_types, bool all_same,
                                                   bool skip_check) {
  std::vector<KernelObjectType> ret;
  //  Use the selected_object_types in the equal size scenario.
  if (object_types.size() == selected_object_types.size()) {
    for (size_t i = 0; i < selected_object_types.size(); ++i) {
      // Allsame/skip_check doesn't support the backoff.
      bool not_backoff = ((all_same || skip_check) && (selected_object_types[i] != object_types[i]));
      if (not_backoff) {
        (void)ret.emplace_back(TypeIdToKernelObjectTypeForTupleUnfold(object_types[i]));
      } else {
        (void)ret.emplace_back(TypeIdToKernelObjectType(selected_object_types[i]));
      }
    }
    return ret;
  }

  // Use the object_types in the unequal size scenario, and convert tuple to tupleUnflod.
  for (size_t i = 0; i < object_types.size(); ++i) {
    (void)ret.emplace_back(TypeIdToKernelObjectTypeForTupleUnfold(object_types[i]));
  }
  return ret;
}

std::vector<KernelObjectType> CalOutputElementObjectTypes(const AnfNodePtr &kernel_node,
                                                          const kernel::KernelAttr &selected_kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_output_object_types = GetOutputObjectTypeListFromKernelAttr(selected_kernel_attr);
  MS_LOG(DEBUG) << "Output object type:" << selected_output_object_types << " for node:" << kernel_node->DebugString()
                << " select attr:" << kernel::FetchPrintInfoByKernelAttr(selected_kernel_attr);
  auto element_num = GetOutputNum(kernel_node);
  if (selected_kernel_attr.GetAllSame() && selected_output_object_types.size() == 1) {
    return std::vector<KernelObjectType>(element_num, TypeIdToKernelObjectType(selected_output_object_types[0]));
  }
  MS_EXCEPTION_IF_CHECK_FAIL(element_num == selected_output_object_types.size(),
                             "Check multi-output kernel attr size failed.");
  return TypeIdToKernelObjectType(selected_output_object_types);
}

std::string FetchPrintInfoByKernelAttr(KernelAttr selected_kernel_attr) {
  std::string attr_info = "input[";
  (void)std::for_each(std::begin(selected_kernel_attr.input_type()), std::end(selected_kernel_attr.input_type()),
                      [&attr_info](auto &input_type) {
                        attr_info += TypeIdToString(input_type.object_type) + " " + TypeIdToString(input_type.dtype) +
                                     " " + input_type.format + ",";
                      });
  attr_info += "] output[";
  (void)std::for_each(std::begin(selected_kernel_attr.output_type()), std::end(selected_kernel_attr.output_type()),
                      [&attr_info](auto &output_type) {
                        attr_info += TypeIdToString(output_type.object_type) + " " + TypeIdToString(output_type.dtype) +
                                     " " + output_type.format + ",";
                      });
  attr_info += "]";
  return attr_info;
}

KernelAttr &KernelAttr::AddInputAttr(const TypeId &object_type, const TypeId &ms_type, const std::string &format) {
  (void)input_type_.emplace_back(DataType(ms_type, format, object_type));
  return *this;
}

KernelAttr &KernelAttr::AddOptionalInputAttr(const TypeId &object_type, const TypeId &ms_type,
                                             const std::string &format) {
  (void)input_type_.emplace_back(DataType(ms_type, format, object_type, true));
  return *this;
}

KernelAttr &KernelAttr::AddOutputAttr(const TypeId &object_type, const TypeId &ms_type, const std::string &format) {
  (void)output_type_.emplace_back(DataType(ms_type, format, object_type));
  return *this;
}

KernelAttr &KernelAttr::AddInputAttr(const TypeId &ms_type, const std::string &format) {
  (void)input_type_.emplace_back(DataType(ms_type, format));
  return *this;
}

KernelAttr &KernelAttr::AddOptionalInputAttr(const TypeId &ms_type, const std::string &format) {
  (void)input_type_.emplace_back(DataType(ms_type, format, kObjectTypeTensorType, true));
  return *this;
}

KernelAttr &KernelAttr::AddOutputAttr(const TypeId &ms_type, const std::string &format) {
  (void)output_type_.emplace_back(DataType(ms_type, format));
  return *this;
}

KernelAttr &KernelAttr::AddAllSameAttr(bool all_same, size_t all_same_input_num, bool group_allsame) {
  all_same_ = all_same;
  is_group_allsame_ = group_allsame;
  if (all_same_input_num < 1) {
    MS_LOG(EXCEPTION) << "Allsame attr must >= 1, but get " << all_same_input_num;
  }
  all_same_input_num_ = all_same_input_num;
  return *this;
}

KernelAttr &KernelAttr::AddSkipCheckAttr(bool skip_check) {
  skip_check_ = skip_check;
  return *this;
}

KernelAttr &KernelAttr::AddRealTuple(const bool &is_real_tuple) {
  is_real_tuple_ = is_real_tuple;
  return *this;
}

KernelAttr &KernelAttr::AddOutInRef(size_t output_index, size_t input_index) {
  out_in_ref_map_[output_index] = input_index;
  return *this;
}

KernelAttr &KernelAttr::AddAllOutInRef(bool all_out_in_ref) {
  all_out_in_ref_ = all_out_in_ref;
  return *this;
}

void KernelAttr::SetInputAttr(const size_t index, const TypeId &ms_type, const std::string &format) {
  if (index >= input_type_.size()) {
    MS_LOG(EXCEPTION) << "Invalid index for input: " << index << ", out of range.";
  }
  input_type_[index] = DataType(ms_type, format);
}

void KernelAttr::SetOutputAttr(const size_t index, const TypeId &ms_type, const std::string &format) {
  if (index >= output_type_.size()) {
    MS_LOG(EXCEPTION) << "Invalid index for output: " << index << ", out of range.";
  }
  output_type_[index] = DataType(ms_type, format);
}

void KernelAttr::SetInputAttrList(const std::vector<DataType> &addr_list) {
  input_type_.assign(addr_list.begin(), addr_list.end());
}

void KernelAttr::SetOutputAttrList(const std::vector<DataType> &addr_list) {
  output_type_.assign(addr_list.begin(), addr_list.end());
}

std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr) {
  std::stringstream ss;
  ss << "[Kernel Attr] all same: " << kernel_attr.GetAllSame();
  if (kernel_attr.GetSkipCheck()) {
    ss << ", skip check: true";
  }
  size_t input_num = kernel_attr.GetInputSize();
  if (input_num > 0) {
    ss << ", input(";
    for (size_t i = 0; i < input_num; ++i) {
      ss << TypeIdLabel(kernel_attr.GetInputAttr(i).dtype);
      if (kernel_attr.GetInputAttr(i).is_optional) {
        ss << "|None";
      }
      if (i != input_num - 1) {
        ss << ",";
      }
    }
    ss << ") ";
  }
  size_t output_num = kernel_attr.GetOutputSize();
  if (output_num > 0) {
    ss << ", output(";
    for (size_t i = 0; i < output_num; ++i) {
      ss << TypeIdLabel(kernel_attr.GetOutputAttr(i).dtype);
      if (i != output_num - 1) {
        ss << ",";
      }
    }
    ss << ").";
  }

  return os << ss.str();
}

bool CheckAttrForAllSameInput(const size_t input_num, const std::vector<mindspore::TypeId> &input_types,
                              const KernelAttr &cur_kernel_attr) {
  auto cur_input_num = cur_kernel_attr.GetInputSize();
  bool is_group_allsame = cur_kernel_attr.GetGroupAllSame();
  size_t cur_all_same_input_num = cur_kernel_attr.GetAllSameInputNum();  // default 0; else >=1 when allsame=true
  size_t cur_standalone_input_num = cur_input_num - cur_all_same_input_num;
  size_t each_attr_input_num =
    (input_num - cur_standalone_input_num) / (cur_all_same_input_num == 0 ? 1 : cur_all_same_input_num);
  // deal with allsame inputs
  if (is_group_allsame) {
    for (size_t i = 0; i < each_attr_input_num; ++i) {
      for (size_t j = 0; j < cur_all_same_input_num; ++j) {
        auto dtype = cur_kernel_attr.GetInputAttr(j).dtype;
        auto start = j + i * cur_all_same_input_num;
        if (input_types[start] != dtype && input_types[start] != kTypeUnknown) {
          return true;
        }
      }
    }
  } else {
    for (size_t i = 0; i < cur_all_same_input_num; ++i) {
      for (size_t j = 0; j < each_attr_input_num; ++j) {
        auto dtype = cur_kernel_attr.GetInputAttr(i).dtype;
        auto start = j + i * each_attr_input_num;
        if (input_types[start] != dtype && input_types[start] != kTypeUnknown) {
          return true;
        }
      }
    }
  }

  // deal with the rest except allsame inputs
  for (size_t i = cur_all_same_input_num; i < cur_standalone_input_num; ++i) {
    auto dtype = cur_kernel_attr.GetInputAttr(i).dtype;
    auto start = each_attr_input_num * cur_all_same_input_num + i;
    if (!(cur_kernel_attr.GetInputAttr(i).is_optional && input_types[start] == kMetaTypeNone) &&
        (input_types[start] != dtype && input_types[start] != kTypeUnknown)) {
      return true;
    }
  }
  return false;
}

std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr,
                                        const std::vector<KernelAttr> &kernel_attr_list) {
  // kernel_attr should not be all same. If so, then return false.
  if (kernel_attr.GetAllSame()) {
    return std::make_pair(false, 0);
  }
  auto input_num = kernel_attr.GetInputSize();
  auto output_num = kernel_attr.GetOutputSize();

  for (size_t index = 0; index < kernel_attr_list.size(); ++index) {
    const auto &cur_kernel_attr = kernel_attr_list[index];
    auto cur_input_num = cur_kernel_attr.GetInputSize();
    auto cur_output_num = cur_kernel_attr.GetOutputSize();
    if (!cur_kernel_attr.GetAllSame() && (input_num != cur_input_num || output_num != cur_output_num)) {
      continue;
    }
    std::vector<mindspore::TypeId> input_types;
    (void)std::transform(kernel_attr.input_type().begin(), kernel_attr.input_type().end(),
                         std::back_inserter(input_types), [](const DataType &Dtype) { return Dtype.dtype; });

    bool mis_match = CheckAttrForAllSameInput(input_num, input_types, cur_kernel_attr);
    if (mis_match) {
      continue;
    }

    for (size_t i = 0; i < output_num; ++i) {
      auto dtype = cur_kernel_attr.GetOutputAttr(cur_kernel_attr.GetAllSame() ? 0 : i).dtype;
      if (kernel_attr.GetOutputAttr(i).dtype != dtype && kernel_attr.GetOutputAttr(i).dtype != kTypeUnknown) {
        mis_match = true;
        break;
      }
    }
    if (!mis_match) {
      return std::make_pair(true, index);
    }
  }

  return std::make_pair(false, 0);
}

static bool IsEquivalentFormat(const std::string &src_format, const std::string &dst_format) {
  if (src_format == dst_format) {
    return true;
  }

  // Equivalent default format.
  if (((src_format == kOpFormat_DEFAULT) || (src_format == kOpFormat_NCHW) || (src_format == kOpFormat_ND)) &&
      ((dst_format == kOpFormat_DEFAULT) || (dst_format == kOpFormat_NCHW) || (dst_format == kOpFormat_ND))) {
    return true;
  }

  return false;
}

std::pair<bool, size_t> MatchKernelAttrStrict(const KernelAttr &kernel_attr,
                                              const std::vector<KernelAttr> &kernel_attr_list) {
  auto input_num = kernel_attr.GetInputSize();
  auto output_num = kernel_attr.GetOutputSize();
  auto AttrMatched = [](const DataType &attr, const DataType &compared_attr) {
    return (attr.dtype != compared_attr.dtype && attr.dtype != kTypeUnknown) ||
           (!IsEquivalentFormat(attr.format, compared_attr.format)) ||
           (attr.object_type != compared_attr.object_type && attr.object_type != kTypeUnknown);
  };
  for (size_t index = 0; index < kernel_attr_list.size(); ++index) {
    const auto &cur_kernel_attr = kernel_attr_list[index];
    // Attr skip indicates that any attr is supported.
    if (cur_kernel_attr.GetSkipCheck()) {
      return std::make_pair(true, index);
    }
    auto cur_input_num = cur_kernel_attr.GetInputSize();
    auto cur_output_num = cur_kernel_attr.GetOutputSize();
    // The num must be equal when not all same.
    if (!cur_kernel_attr.GetAllSame() && (input_num != cur_input_num || output_num != cur_output_num)) {
      continue;
    }

    bool mis_match = false;
    // Check the input attrs.
    for (size_t i = 0; i < cur_input_num; ++i) {
      MS_EXCEPTION_IF_CHECK_FAIL((kernel_attr.GetInputSize() > i), "The input num is out of range.");
      auto &input_attr = kernel_attr.GetInputAttr(i);
      auto &cur_input_attr = cur_kernel_attr.GetInputAttr(i);
      if (AttrMatched(input_attr, cur_input_attr)) {
        mis_match = true;
        break;
      }
    }

    if (mis_match) {
      continue;
    }

    // Check the output attrs.
    for (size_t i = 0; i < cur_output_num; ++i) {
      MS_EXCEPTION_IF_CHECK_FAIL((kernel_attr.GetOutputSize() > i), "The output num is out of range.");
      auto &output_attr = kernel_attr.GetOutputAttr(i);
      auto &cur_output_attr = cur_kernel_attr.GetOutputAttr(i);
      if (AttrMatched(output_attr, cur_output_attr)) {
        mis_match = true;
        break;
      }
    }

    if (!mis_match) {
      return std::make_pair(true, index);
    }
  }

  return std::make_pair(false, 0);
}

std::pair<std::vector<DataType>, std::vector<DataType>> GetInOutDataTypesFromKernelAttr(const KernelAttr &kernel_attr) {
  size_t input_attr_size = kernel_attr.GetInputSize();
  std::vector<DataType> input_data_types;
  for (size_t i = 0; i < input_attr_size; ++i) {
    input_data_types.push_back(kernel_attr.GetInputAttr(i));
  }

  size_t output_attr_size = kernel_attr.GetOutputSize();
  std::vector<DataType> output_data_types;
  for (size_t i = 0; i < output_attr_size; ++i) {
    output_data_types.push_back(kernel_attr.GetOutputAttr(i));
  }

  return std::make_pair(input_data_types, output_data_types);
}

bool IsFoldKernelBuildInfo(const KernelBuildInfoPtr &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  auto inputs_object_type = kernel_build_info->GetAllInputKernelObjectTypes();
  if (std::find(inputs_object_type.begin(), inputs_object_type.end(), KernelObjectType::TUPLE) !=
      inputs_object_type.end()) {
    return true;
  }

  auto outputs_object_type = kernel_build_info->GetAllOutputKernelObjectTypes();
  if (std::find(outputs_object_type.begin(), outputs_object_type.end(), KernelObjectType::TUPLE) !=
      outputs_object_type.end()) {
    return true;
  }

  return false;
}

KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info) {
  MS_EXCEPTION_IF_NULL(build_info);
  KernelAttr kernel_attr;
  for (size_t i = 0; i < build_info->GetInputNum(); ++i) {
    (void)kernel_attr.AddInputAttr(KernelObjectTypeToTypeId(build_info->GetInputKernelObjectType(i)),
                                   build_info->GetInputDeviceType(i), build_info->GetInputFormat(i));
  }
  for (size_t j = 0; j < build_info->GetOutputNum(); ++j) {
    (void)kernel_attr.AddOutputAttr(KernelObjectTypeToTypeId(build_info->GetOutputKernelObjectType(j)),
                                    build_info->GetOutputDeviceType(j), build_info->GetOutputFormat(j));
  }
  return kernel_attr;
}

KernelAttr GetKernelAttrFromTensors(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  KernelAttr kernel_attr;
  for (auto tensor : inputs) {
    (void)kernel_attr.AddInputAttr(tensor->dtype_id(), GetFormatFromEnumToStr(tensor->format()));
  }
  for (auto tensor : outputs) {
    (void)kernel_attr.AddOutputAttr(tensor->dtype_id(), GetFormatFromEnumToStr(tensor->format()));
  }
  return kernel_attr;
}

void SyncOutInRef(const KernelAttr &from_kernel_attr, KernelAttr *to_kernel_attr) {
  const auto &out_in_ref = from_kernel_attr.GetOutInRefMap();
  bool all_out_in_ref = from_kernel_attr.GetAllOutInRef();
  for (const auto &ref : out_in_ref) {
    (void)to_kernel_attr->AddOutInRef(ref.first, ref.second);
  }
  (void)to_kernel_attr->AddAllOutInRef(all_out_in_ref);
}

namespace math {
void SinCosf(float x, float *sinv, float *cosv) {
  *sinv = sinf(x);
  *cosv = cosf(x);
}
}  // namespace math
}  // namespace kernel
}  // namespace mindspore
