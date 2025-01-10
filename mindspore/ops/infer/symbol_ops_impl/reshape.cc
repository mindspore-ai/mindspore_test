/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/infer/symbol_ops_impl/reshape.h"
#include <set>
#include <memory>
#include <utility>
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"
#include "mindspore/core/include/utils/ordered_set.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
std::string Reshape::DumpText() const {
  std::ostringstream oss;
  oss << InferShapeOp::DumpText();
  for (auto &inner_op : inner_ops_) {
    oss << "  " << inner_op->DumpText();
  }
  return oss.str();
}

size_t Reshape::FindUnknownDim(const SymbolPtrList &symbols) {
  size_t unknown_dim_idx = symbols.size();
  size_t notpositive_symbol_cnt = 0;
  size_t notpositive_symbol_idx = 0;
  for (size_t i = 0; i < symbols.size(); i++) {
    auto item = symbols[i]->as<IntSymbol>();
    if (item->is_negative()) {
      unknown_dim_idx = i;
      break;
    } else if (!item->is_greater_equal(0)) {
      // unknown positive or negative
      notpositive_symbol_cnt++;
      notpositive_symbol_idx = i;
    }
  }

  if (unknown_dim_idx < symbols.size()) {
    return unknown_dim_idx;
  }
  // when there is no a certainly negative symbol, and there is only one not-positive symbol, set it to unknown.
  if (notpositive_symbol_cnt == 1) {
    return notpositive_symbol_idx;
  }
  return symbols.size();
}

void Reshape::RemoveSameSymbol(SymbolPtrList *inp_symbols, SymbolPtrList *out_symbols) {
  OrderedSet<SymbolPtr> input(*inp_symbols);
  OrderedSet<SymbolPtr> output(*out_symbols);
  for (auto it1 = input.begin(); it1 != input.end();) {
    bool removed = false;
    for (auto it2 = output.begin(); it2 != output.end();) {
      if ((*it1)->EqualsTo(*it2)) {
        removed = true;
        it1 = input.erase(it1);
        it2 = output.erase(it2);
        break;
      }
      ++it2;
    }
    if (!removed) {
      ++it1;
    }
  }
  if (input.size() < inp_symbols->size()) {
    *inp_symbols = SymbolPtrList(input.begin(), input.end());
    *out_symbols = SymbolPtrList(output.begin(), output.end());
  }
}

SymbolPtr Reshape::Eval() {
  // only eval on Building
  auto data = input_as<ListSymbol>(0);
  auto shape = input_as<ListSymbol>(1);
  if (!shape->HasData()) {
    return GenVList();
  }
  if (std::all_of(shape->symbols().begin(), shape->symbols().end(),
                  [](const SymbolPtr &s) { return s->as<IntSymbol>()->is_greater_equal(0); })) {
    // all items of "shape" are positive,
    DoNotEvalOnRun();
    return input(1);
  }
  auto inp_symbols = data->symbols();
  auto out_symbols = shape->symbols();
  RemoveSameSymbol(&inp_symbols, &out_symbols);
  if (out_symbols.empty()) {
    // all output symbols are from input.
    DoNotEvalOnRun();
    return input(1);
  }
  // do not add the inner operation to global op list.
  OperationEmitter e(&inner_ops_);
  SetEmitter(&e);

  auto unknown_dim_idx = FindUnknownDim(out_symbols);
  SymbolPtrList result = shape->symbols();
  if (unknown_dim_idx == out_symbols.size()) {
    for (size_t i = 0; i < result.size(); i++) {
      if (!result[i]->as<IntSymbol>()->is_greater_equal(0)) {
        result[i] = GenVInt();
      }
    }
    return GenList(std::move(result));
  }
  auto unknown_dim_symbol = out_symbols[unknown_dim_idx];
  out_symbols.erase(out_symbols.begin() + unknown_dim_idx);
  unknown_dim_idx = static_cast<size_t>(std::find(result.begin(), result.end(), unknown_dim_symbol) - result.begin());
  if (unknown_dim_idx >= result.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The symbol " << unknown_dim_symbol->ToString() << " is not found in output shape";
  }

  // "-1" exists in shape.
  if (data->is_dyn_len()) {
    result[unknown_dim_idx] = GenVInt();
    return GenList(std::move(result));
  }
  SymbolPtr input_unknown_dims = nullptr;
  SymbolPtr output_unknown_dims = nullptr;
  int64_t input_const_dims = 1;
  int64_t output_const_dims = 1;
  Accumulate<ScalarMul>(inp_symbols, emitter(), &input_unknown_dims, &input_const_dims);
  Accumulate<ScalarMul>(out_symbols, emitter(), &output_unknown_dims, &output_const_dims);
  if (input_const_dims != 1 && output_const_dims != 1) {
    auto g = std::gcd(input_const_dims, output_const_dims);
    input_const_dims /= g;
    output_const_dims /= g;
  }
  auto tmp1 = input_unknown_dims != nullptr
                ? Emit(std::make_shared<ScalarMul>(input_unknown_dims, GenInt(input_const_dims)))
                : GenInt(input_const_dims);
  auto tmp2 = output_unknown_dims != nullptr
                ? Emit(std::make_shared<ScalarMul>(output_unknown_dims, GenInt(output_const_dims)))
                : GenInt(output_const_dims);
  result[unknown_dim_idx] = Emit(std::make_shared<ScalarDiv>(tmp1, tmp2));
  return GenList(std::move(result));
}

void Reshape::EvalOnRun() {
  auto shape = input_as<ListSymbol>(kIndex1);
  size_t unknown_dim_idx = shape->size();
  int64_t output_const_dims = 1LL;
  for (size_t i = 0; i < shape->size(); i++) {
    auto v = shape->item_as<IntSymbol>(i)->value();
    if (v <= 0) {
      if (unknown_dim_idx != shape->size()) {
        MS_EXCEPTION(ValueError) << "For 'Reshape', there are more than one \"-1\" in \"shape\": " << shape->ToString();
      }
      unknown_dim_idx = i;
    } else {
      output_const_dims *= v;
    }
  }
  if (unknown_dim_idx == shape->size()) {
    // no "-1" in "shape"
    output_->Update(input(1));
    return;
  }
  SymbolPtr data_sym = nullptr;
  int64_t data_size = 1;
  auto data = input_as<ListSymbol>(kIndex0);
  Accumulate<ScalarMul>(data->symbols(), emitter(), &data_sym, &data_size);
  if (data_sym != nullptr) {
    MS_EXCEPTION(ValueError) << "For 'Reshape', the input shape has dynamic dim in runtime: " << data->ToString();
  }
  // the size of "-1"
  auto result = shape->symbols();
  if (data_size % output_const_dims != 0) {
    MS_EXCEPTION(ValueError) << "For 'Reshape', the input " << data->ToString() << " can not be reshape to "
                             << shape->ToString();
  }
  result[unknown_dim_idx] = GenInt(data_size / output_const_dims);
  output_as<ListSymbol>()->UpdateList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Reshape").SetShapeDepend({DependOn::kShape, DependOn::kValue}).SetShapeFuncWith<Reshape>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
