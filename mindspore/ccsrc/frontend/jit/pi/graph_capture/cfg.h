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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_CFG_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_CFG_H

#include <memory>
#include <set>
#include <map>
#include <string>
#include <vector>
#include "frontend/jit/pi/python_adapter/py_code.h"
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/graph_capture/local_liveness.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

class CFG;
class Block;
class Instr {
 public:
  Instr &operator=(const Instr &) = delete;
  Instr(const Instr &o)
      : bci_(-1), op_(o.op_), arg_(o.arg_), name_(o.name_), cnst_(o.cnst_), jump_(nullptr), loc_(o.loc_) {}
  Instr(int op, int arg) : bci_(-1), op_(op), arg_(arg), jump_(nullptr), loc_{-1, -1, -1, -1} {}
  Instr(int op, int arg, const CodeLocation &loc) : Instr(op, arg) { loc_ = loc; }
  Instr(int op, int arg, const std::string &name) : Instr(op, arg) { name_ = name; }
  Instr(int op, int arg, const py::object &cnst) : Instr(op, arg) { cnst_ = cnst; }
  Instr(int op, int arg, int bci, int line = -1) : Instr(op, arg) { bci_ = bci, loc_ = {line, line, -1, -1}; }
  explicit Instr(int op) : Instr(op, 0) {}

  int bci() const { return bci_; }
  void set_bci(int i) { bci_ = i; }
  int op() const { return op_; }
  void set_op(int op) { op_ = op; }
  int arg() const { return arg_; }
  void set_arg(int arg) { arg_ = arg; }
  int line() const { return loc_.start_line_; }
  void set_line(int l) { loc_.start_line_ = l; }
  Instr *extra_jump() const { return jump_; }
  void set_extra_jump(Instr *j) { jump_ = j; }
  const auto &location() const { return loc_; }
  void set_location(const CodeLocation &loc) { loc_ = loc; }

  const std::string &name() const { return name_; }
  void set_name(const std::string &n) { name_ = n; }
  void set_name(const char *n) { name_ = n ? n : ""; }
  const py::object &cnst() const { return cnst_; }
  void set_cnst(const py::handle &cnst) { cnst_ = py::reinterpret_borrow<py::object>(cnst); }

  int InstrSize() const;
  std::string ToString() const;

 private:
  int bci_;
  int op_;
  int arg_;
  // these field only one is valid, union this these field like this { const char *, PyObject *, Instr * } ?
  std::string name_;
  // if python3.11 ~ python3.12 and opcode is call, `cnst_` is KW_NAMES
  py::object cnst_;
  Instr *jump_;
  CodeLocation loc_;
};

class Block {
 public:
  uint32_t id() const { return id_; }
  void set_id(uint32_t arg) { id_ = arg; }
  const auto &pred_bbs() const { return pred_bbs_; }
  const auto &succ_bbs() const { return succ_bbs_; }
  int begin_ci() const { return begin_; }
  int end_ci() const { return end_; }
  void set_begin_ci(int i) { begin_ = i; }
  void set_end_ci(int i) { end_ = i; }

  void set_is_loop_head(bool flag) {
    is_loop_head_ = flag;
    if (flag) {
      loop_body_bbs_.insert(this);
      loop_head_bb_ = this;
    }
  }
  void add_loop_body(Block *block) {
    loop_body_bbs_.insert(block);
    block->set_loop_head_bb(this);
  }
  const std::set<Block *> &loop_body_bbs() const { return loop_body_bbs_; }
  Block *loop_head_bb() const { return loop_head_bb_; }
  void set_loop_head_bb(Block *block) { loop_head_bb_ = block; }
  bool is_loop_head() const { return is_loop_head_; }
  void set_is_loop_body(bool flag) { is_loop_body_ = flag; }
  bool is_loop_body() const { return is_loop_body_; }
  bool is_dead() const { return is_dead_; }
  void set_is_dead(bool flag) { is_dead_ = flag; }

  std::string Dump(bool dump_instr = true) const;
  void AddSuccBB(Block *bb);
  void set_loop_head(Block *pBlock);
  std::string ToString() const { return Dump(false); }

 private:
  uint32_t id_;  // start from 0
  int begin_;
  int end_;
  std::set<Block *> pred_bbs_;
  std::set<Block *> succ_bbs_;
  // if curr bb is loop head, loop_body_bbs_ will include all loop body bbs
  std::set<Block *> loop_body_bbs_;
  Block *loop_head_bb_ = nullptr;

  bool is_loop_body_ = false;
  bool is_loop_head_ = false;
  bool is_dead_ = true;
};

class CFG {
 public:
  explicit CFG(PyCodeObject *co) : co_(co) {}

  class BBIterator {
   public:
    BBIterator() = default;
    explicit BBIterator(const CFG *c) : visit_(c->bb_pool().size(), false) {
      q_.push_back(c->GetFirstBB());
      visit_[c->GetFirstBB()->id()] = true;
    }

    BBIterator(const CFG *c, Block *bb) : visit_(c->bb_pool().size(), false) {
      q_.push_back(bb);
      visit_[bb->id()] = true;
    }

    const auto &GetVisitMap() const { return visit_; }
    Block *operator*() const { return q_.front(); }
    bool operator!=(const BBIterator &end) const { return !q_.empty(); }
    BBIterator &operator++();

    std::vector<Block *> q_;
    std::vector<bool> visit_;
  };

  BBIterator begin() const { return BBIterator(this); }
  BBIterator begin(Block *start) const { return BBIterator(this, start); }
  BBIterator end() const { return BBIterator(); }

  const std::vector<std::unique_ptr<Block>> &bb_pool() const { return bb_pool_; }
  const std::vector<std::unique_ptr<Instr>> &instr_pool() const { return instrs_; }
  const std::unique_ptr<Liveness> &liveness() const { return liveness_; }
  std::vector<std::unique_ptr<Instr>> &instr_pool() { return instrs_; }
  std::vector<std::unique_ptr<Block>> &bb_pool() { return bb_pool_; }
  std::unique_ptr<Liveness> &liveness() { return liveness_; }
  const ExceptionTable &exc_table() const { return exc_table_; }
  // python3.11+ only, find first exception table item of try/with blocks
  ExceptionTable::const_iterator FindTryWithBlock(int bci) const;
  ExceptionTable::const_iterator FindExcTableItem(int bci) const;
  int GetLocalCount() const { return co_.LocalSize(); }

  const Liveness *GetLiveness();

  void GenerateCFG();
  void MarkDeadBB();

  Block *GetFirstBB() const { return bb_pool_.size() ? bb_pool_[0].get() : nullptr; }
  Block *GetBlockByBci(int) const;
  Instr *GetBlockTail(Block *) const;

  std::string ToString() const;

 private:
  Instr *GetInstruction(int bci);
  void BuildInst(const uint8_t *begin, const uint8_t *end);
  std::map<int, Block *> BuildBB(const uint8_t *begin, const uint8_t *end);
  void BuildCFG(const std::map<int, Block *> &labels);
  ExceptionTable::const_iterator FindTryWithStart(ExceptionTable::const_iterator iter) const;

  PyCodeWrapper co_;
  std::vector<std::unique_ptr<Instr>> instrs_;
  std::vector<std::unique_ptr<Block>> bb_pool_;
  std::unique_ptr<Liveness> liveness_;
  ExceptionTable exc_table_;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_CFG_H
