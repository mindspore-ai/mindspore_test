/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/utils/opcode_util.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "frontend/jit/pi/utils/opcode_declare.h"

namespace mindspore {
namespace pijit {

Opcode Opcode::opmap[Opcode::kMaxCode];
const Opcode Opcode::k_ILLEGAL_OPCODE = {"ILLEGAL_OPCODE", ILLEGAL_OPCODE, Opcode::Class::kNop, 0};

constexpr const char *binary_enum_map[] = {
  "+",    // NB_ADD                                   0
  "&",    // NB_AND                                   1
  "//",   // NB_FLOOR_DIVIDE                          2
  "<<",   // NB_LSHIFT                                3
  "@",    // NB_MATRIX_MULTIPLY                       4
  "*",    // NB_MULTIPLY                              5
  "%",    // NB_REMAINDER                             6
  "|",    // NB_OR                                    7
  "**",   // NB_POWER                                 8
  ">>",   // NB_RSHIFT                                9
  "-",    // NB_SUBTRACT                             10
  "/",    // NB_TRUE_DIVIDE                          11
  "^",    // NB_XOR                                  12
  "+=",   // NB_INPLACE_ADD                          13
  "&=",   // NB_INPLACE_AND                          14
  "//=",  // NB_INPLACE_FLOOR_DIVIDE                 15
  "<<=",  // NB_INPLACE_LSHIFT                       16
  "@=",   // NB_INPLACE_MATRIX_MULTIPLY              17
  "*=",   // NB_INPLACE_MULTIPLY                     18
  "%=",   // NB_INPLACE_REMAINDER                    19
  "|=",   // NB_INPLACE_OR                           20
  "**=",  // NB_INPLACE_POWER                        21
  ">>=",  // NB_INPLACE_RSHIFT                       22
  "-=",   // NB_INPLACE_SUBTRACT                     23
  "/=",   // NB_INPLACE_TRUE_DIVIDE                  24
  "^=",   // NB_INPLACE_XOR                          25
};

constexpr const char *compare_enum_map[] = {
  "<",   // Py_LT 0
  "<=",  // Py_LE 1
  "==",  // Py_EQ 2
  "!=",  // Py_NE 3
  ">",   // Py_GT 4
  ">=",  // Py_GE 5
};

static auto BinaryOpMapEnum() {
  static uint8_t nb_map[256] = {0};
  nb_map[BINARY_ADD] = 0;
  nb_map[BINARY_AND] = 1;
  nb_map[BINARY_FLOOR_DIVIDE] = 2;
  nb_map[BINARY_LSHIFT] = 3;
  nb_map[BINARY_MATRIX_MULTIPLY] = 4;
  nb_map[BINARY_MULTIPLY] = 5;
  nb_map[BINARY_MODULO] = 6;
  nb_map[BINARY_OR] = 7;
  nb_map[BINARY_POWER] = 8;
  nb_map[BINARY_RSHIFT] = 9;
  nb_map[BINARY_SUBTRACT] = 10;
  nb_map[BINARY_TRUE_DIVIDE] = 11;
  nb_map[BINARY_XOR] = 12;
  nb_map[INPLACE_ADD] = 13;
  nb_map[INPLACE_AND] = 14;
  nb_map[INPLACE_FLOOR_DIVIDE] = 15;
  nb_map[INPLACE_LSHIFT] = 16;
  nb_map[INPLACE_MATRIX_MULTIPLY] = 17;
  nb_map[INPLACE_MULTIPLY] = 18;
  nb_map[INPLACE_MODULO] = 19;
  nb_map[INPLACE_OR] = 20;
  nb_map[INPLACE_POWER] = 21;
  nb_map[INPLACE_RSHIFT] = 22;
  nb_map[INPLACE_SUBTRACT] = 23;
  nb_map[INPLACE_TRUE_DIVIDE] = 24;
  nb_map[INPLACE_XOR] = 25;
  return nb_map;
}

enum OpcodeFlag {
  kJRel = 1 << 0,      // is jump relative
  kJAbs = 1 << 1,      // is jump relative
  kNotFall = 1 << 2,   // jump directly, return, raise
  kHasConst = 1 << 3,  // has const in co_consts
  kHasName = 1 << 4,   // has name in co_names
  kHasFree = 1 << 5,   // has free variable operations, not is 'free' of this function
  kCanDel = 1 << 6,    // can be remove if result is unused
  /**
   * Maybe remove if result is unused.
   * Generally or literally, it's no side effect, check it and parse
   * all user-defined operation to call function while graph building
   */
  kMayDel = 1 << 7,
  kJBak = 1 << 8,
};

Opcode::Opcode() { *this = k_ILLEGAL_OPCODE; }

bool Opcode::IsJRel() const { return flag_ & kJRel; }
bool Opcode::IsJAbs() const { return flag_ & kJAbs; }
bool Opcode::IsJBack() const { return flag_ & kJBak; }
bool Opcode::IsNotFall() const { return flag_ & kNotFall; }
bool Opcode::HasName() const { return flag_ & kHasName; }
bool Opcode::HasFree() const { return flag_ & kHasFree; }
bool Opcode::HasConst() const { return flag_ & kHasConst; }
bool Opcode::CanDelete(int oparg) const { return (flag_ & kCanDel) || CheckIsOp(oparg); }
bool Opcode::MayDelete(int oparg) const { return (flag_ & kMayDel) || CanDelete(oparg); }
bool Opcode::IsExcMatch(int oparg) const {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  return oparg == PyCmp_EXC_MATCH;
#else
  return false;
#endif
}

bool Opcode::IsCallFunc() const {
  if (code_ == ILLEGAL_OPCODE) {
    return false;
  }
  return code_ == CALL || code_ == CALL_FUNCTION;
}

bool Opcode::IsConditionJump() const {
  if (code_ == ILLEGAL_OPCODE) {
    return false;
  }
  return code_ == POP_JUMP_BACKWARD_IF_FALSE || code_ == POP_JUMP_BACKWARD_IF_NONE ||
         code_ == POP_JUMP_BACKWARD_IF_NOT_NONE || code_ == POP_JUMP_BACKWARD_IF_TRUE ||
         code_ == POP_JUMP_FORWARD_IF_FALSE || code_ == POP_JUMP_FORWARD_IF_NONE ||
         code_ == POP_JUMP_FORWARD_IF_NOT_NONE || code_ == POP_JUMP_FORWARD_IF_TRUE || code_ == POP_JUMP_IF_FALSE ||
         code_ == POP_JUMP_IF_TRUE || code_ == JUMP_IF_FALSE_OR_POP || code_ == JUMP_IF_TRUE_OR_POP;
}

bool Opcode::IsBoolConditionJump() const {
  if (code_ == ILLEGAL_OPCODE) {
    return false;
  }
  return code_ == POP_JUMP_BACKWARD_IF_FALSE || code_ == POP_JUMP_BACKWARD_IF_TRUE ||
         code_ == POP_JUMP_FORWARD_IF_FALSE || code_ == POP_JUMP_FORWARD_IF_TRUE || code_ == POP_JUMP_IF_FALSE ||
         code_ == POP_JUMP_IF_TRUE || code_ == JUMP_IF_FALSE_OR_POP || code_ == JUMP_IF_TRUE_OR_POP;
}

bool Opcode::IsNoneConditionJump() const {
  if (code_ == ILLEGAL_OPCODE) {
    return false;
  }
  return code_ == POP_JUMP_BACKWARD_IF_NONE || code_ == POP_JUMP_BACKWARD_IF_NOT_NONE ||
         code_ == POP_JUMP_FORWARD_IF_NONE || code_ == POP_JUMP_FORWARD_IF_NOT_NONE;
}

// see "${PythonInclude}/internal/pycore_opcode.h"
static uint8_t *GetOpCacheCount() {
  static uint8_t cache[256] = {0};  // memset to zero
#if IS_PYTHON_3_13_PLUS
  // #error "Not implement for python3.13 opcode");
#elif IS_PYTHON_3_12_PLUS
  cache[BINARY_SUBSCR] = 1;
  cache[STORE_SUBSCR] = 1;
  cache[UNPACK_SEQUENCE] = 1;
  cache[FOR_ITER] = 1;
  cache[STORE_ATTR] = 4;
  cache[LOAD_ATTR] = 9;
  cache[COMPARE_OP] = 1;
  cache[LOAD_GLOBAL] = 4;
  cache[BINARY_OP] = 1;
  cache[SEND] = 1;
  cache[LOAD_SUPER_ATTR] = 1;
  cache[CALL] = 3;
#elif IS_PYTHON_3_11_PLUS
  cache[BINARY_SUBSCR] = 4;
  cache[STORE_SUBSCR] = 1;
  cache[UNPACK_SEQUENCE] = 1;
  cache[STORE_ATTR] = 4;
  cache[LOAD_ATTR] = 4;
  cache[COMPARE_OP] = 2;
  cache[LOAD_GLOBAL] = 5;
  cache[BINARY_OP] = 1;
  cache[LOAD_METHOD] = 10;
  cache[PRECALL] = 1;
  cache[CALL] = 4;
#endif
  return cache;
}

int Opcode::InstrSize(int arg) const {
  static uint8_t *cache = GetOpCacheCount();
  int extended_args = (arg > 0xffffff) + (arg > 0xffff) + (arg > 0xff);
  return extended_args + 1 + cache[code_];
}

const char *Opcode::BinaryMathString(int i) {
  static auto op_to_enum = BinaryOpMapEnum();
  uint8_t arg = static_cast<uint8_t>(i);
  if (code_ == COMPARE_OP) {
    return arg <= Py_GE ? compare_enum_map[arg] : "?";
  }
  uint8_t index = IS_PYTHON_3_11_PLUS ? arg : op_to_enum[code_];
  constexpr int binary_enum_max = sizeof(binary_enum_map) / sizeof(binary_enum_map[0]);
  return index < binary_enum_max ? binary_enum_map[index] : "?";
}

bool Opcode::CheckIsOp(int oparg, bool *invert) const {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  if (invert != nullptr) {
    *invert = oparg == PyCmp_IS_NOT;
  }
  return code_ == COMPARE_OP ? (oparg == PyCmp_IS || oparg == PyCmp_IS_NOT) : false;
#else
  if (invert != nullptr) {
    *invert = oparg;
  }
  return code_ == IS_OP;
#endif
}
bool Opcode::CheckContainsOp(int oparg, bool *invert) const {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  if (invert != nullptr) {
    *invert = oparg == PyCmp_NOT_IN;
  }
  return code_ == COMPARE_OP ? (oparg == PyCmp_IN || oparg == PyCmp_NOT_IN) : false;
#else
  if (invert != nullptr) {
    *invert = oparg;
  }
  return code_ == CONTAINS_OP;
#endif
}

bool Opcode::HasArg() const { return HAS_ARG(code_); }

const Opcode *Opcode::Map() {
  static bool init = false;
  if (init) {
    return opmap;
  }
  init = true;

#define DEF_OPCODE(name, cls, flag) \
  opmap[(name)] = (name) == ILLEGAL_OPCODE ? Opcode::k_ILLEGAL_OPCODE : Opcode(#name, (name), (cls), (flag));

#include "./opcode_attr.def"
#undef DEF_OPCODE

  return opmap;
}

int Opcode::JumpTarget(int pc, int off) const {
  constexpr int mul = IS_PYTHON_3_10_PLUS ? 1 : 2;

  if (IsJRel()) {
    if (IsJBack()) {
      off = -off;
    }
    int tar = pc + InstrSize() + off / mul;
    return tar;
  }
  if (IsJAbs()) {
    int tar = off;
    return tar / mul;
  }
  return -1;
}

int Opcode::JumpOffset(int pc, int tar) const {
  constexpr int mul = IS_PYTHON_3_10_PLUS ? 1 : 2;

  if (IsJRel()) {
    int off = (tar - pc - InstrSize()) * mul;
    if (IsJBack()) {
      // assert tar < offset
      off = -off;
    }
    return off;
  }
  if (IsJAbs()) {
    int off = tar;
    return off * mul;
  }
  return -1;
}

}  // namespace pijit
}  // namespace mindspore
