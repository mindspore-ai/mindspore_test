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
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/utils/opcode_declare.h"

namespace mindspore {
namespace pijit {

constexpr const int PY_BCSIZE = sizeof(_Py_CODEUNIT);

int Instr::InstrSize() const { return Opcode(op_).InstrSize(arg_); }

std::string Instr::ToString() const {
#if IS_PYTHON_3_11_PLUS
  if (op() == CACHE) {
    return "<CACHE>";
  }
#endif
  std::stringstream s;
  s << bci_ << ' ' << Opcode(op_).name() << ' ' << arg_;
  if (!name().empty()) {
    s << "  " << name();
  }
  if (cnst().ptr()) {
    s << "  " << std::string(py::str(cnst().ptr()));
  }
  if (extra_jump()) {
    s << " -> " << extra_jump()->bci();
  }
  return s.str();
}

void Block::AddSuccBB(Block *bb) {
  succ_bbs_.insert(bb);
  bb->pred_bbs_.insert(this);
}

std::string Block::Dump(bool dump_instr) const {
  std::stringstream os;
  os << "Block [" << (begin_ci() * PY_BCSIZE) << ',' << (end_ci() * PY_BCSIZE) << "), (id=" << id_
     << ", is_dead=" << is_dead_ << ", is_loop_head=" << is_loop_head_ << ", is_loop_body_=" << is_loop_body_
     << ", preds={";
  for (Block *bb : pred_bbs_) {
    os << bb->id() << " ";
  }
  os << "}, succs={";
  for (Block *bb : succ_bbs_) {
    os << bb->id() << " ";
  }
  os << "}";
  if (is_loop_head()) {
    os << ", loop_body={";
    for (auto bb : loop_body_bbs_) {
      os << bb->id() << " ";
    }
    os << "}";
  }
  os << ")";
  return os.str();
}

void Block::set_loop_head(Block *block) {
  if (block && block->is_loop_head()) {
    block->add_loop_body(this);
    loop_head_bb_ = block;
  }
}

void CFG::GenerateCFG() {
  if (bb_pool().size() != 0) {
    return;
  }
  py::object code_bytes = co_.Code();
  MS_EXCEPTION_IF_CHECK_FAIL(code_bytes.ptr() != nullptr && PyBytes_Check(code_bytes.ptr()),
                             "system error, code.co_code not bytes");
  const char *bytes = PyBytes_AS_STRING(code_bytes.ptr());
  Py_ssize_t size = PyBytes_Size(code_bytes.ptr());
  const uint8_t *begin = reinterpret_cast<const uint8_t *>(bytes);
  const uint8_t *end = reinterpret_cast<const uint8_t *>(bytes + size);
  this->instrs_.resize(size / PY_BCSIZE);

  BuildInst(begin, end);
  BuildCFG(BuildBB(begin, end));
  MarkDeadBB();
#if IS_PYTHON_3_11_PLUS
  for (size_t index = 0, size = instrs_.size(); index < size; ++index) {
    // code check: std::replace can't apply to std::make_unique
    (void)(instrs_[index] == nullptr ? !(instrs_[index] = std::make_unique<Instr>(CACHE, 0, index)) : false);
  }
#endif
}

// Skip inline CACHE entries
template <typename Ft = void (*)(int off, int op, int arg)>
inline void DecodeInstructionBytes(const uint8_t *begin, const uint8_t *end, Ft yield) {
  const int uint8_w = 8;
  const uint8_t *code = begin;
  int extended_arg = 0;
  int caches = 0;
  int arg;
  for (int i = 0; (begin + i) < end; i += PY_BCSIZE) {
    if (caches) {
      caches--;
      continue;
    }
    int op = code[i];
    // for python3.11+, the bytes from getattr(code, "co_code"), all opcode is de-optimized
    Opcode deop(op);
    caches = deop.InstrSize() - 1;
    arg = code[i + 1] | extended_arg;
    extended_arg = (deop == EXTENDED_ARG) ? (arg << uint8_w) : 0;
    yield(i, deop, arg);
  }
}

void CFG::BuildInst(const uint8_t *begin, const uint8_t *end) {
  const auto make_instr = [this](int off, int op, int arg) {
#if IS_PYTHON_3_11_PLUS
    op = op == LOAD_METHOD ? LOAD_ATTR : op;
#else
    op = op == LOAD_METHOD ? LOAD_ATTR : (op == CALL_METHOD ? CALL_FUNCTION : op);
#endif
    Opcode opcode(op);
    int bci = off / PY_BCSIZE;
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(bci) < instrs_.size(), "Error byte code end");

    int line = PyCode_Addr2Line(co_.ptr(), off);
    if (instrs_[bci] == nullptr) {
      instrs_[bci] = std::make_unique<Instr>(op);
    }
    const auto &cur = instrs_[bci];
    cur->set_bci(bci);
    cur->set_op(op);
    cur->set_arg(arg);
    cur->set_line(line);
    if (opcode.HasConst()) {  // KW_NAMES, LOAD_CONST, RETURN_CONST
      cur->set_cnst(co_.co_consts()[arg]);
    }
    if (opcode.HasName()) {
      int index = arg;
#if IS_PYTHON_3_12_PLUS
      index = op == LOAD_ATTR ? (index >> 1) : index;
#elif IS_PYTHON_3_11_PLUS
      index = op == LOAD_GLOBAL ? (index >> 1) : index;
#endif
      cur->set_name(PyUnicode_AsUTF8(co_.co_names()[index].ptr()));
    }
    if (!opcode.HasJump()) {
      return;
    }
    int jump = opcode.JumpTarget(bci, arg);
    if (instrs_[jump] == nullptr) {
      instrs_[jump] = std::make_unique<Instr>(op);
    }
    cur->set_extra_jump(instrs_[jump].get());
  };

  DecodeInstructionBytes(begin, end, make_instr);
}

std::map<int, Block *> CFG::BuildBB(const uint8_t *begin, const uint8_t *end) {
  std::map<int, Block *> labels;  // ordered map by bci
  const int end_bci = (end - begin) / PY_BCSIZE;

  const auto make_block = [this, &labels, &end_bci](int off, int op, int arg) {
    int bci = off / PY_BCSIZE;
    Opcode opcode(op);
    if (opcode.HasJump()) {
      labels[opcode.JumpTarget(bci, arg)] = nullptr;
    }
    int fall_to = bci + opcode.InstrSize();
    if (fall_to < end_bci && (opcode.IsNotFall() || opcode.HasJump())) {
      labels[fall_to] = nullptr;
    }
  };
  labels[0] = nullptr;
  DecodeInstructionBytes(begin, end, make_block);

  for (auto iter = labels.begin(); iter != labels.end();) {
    size_t id = this->bb_pool_.size();
    bb_pool_.push_back(std::make_unique<Block>());
    iter->second = bb_pool_.back().get();
    Block *cur = iter->second;
    cur->set_id(id);
    cur->set_begin_ci(iter->first);
    ++iter;
    cur->set_end_ci(iter != labels.end() ? iter->first : end_bci);
  }
  return labels;
}

Instr *CFG::GetBlockTail(Block *blk) const {
#if IS_PYTHON_3_11_PLUS
  Instr *instr_tail = nullptr;
  for (int bci = blk->end_ci() - 1; bci >= blk->begin_ci() && (instr_tail == nullptr || instr_tail->op() == CACHE);
       --bci) {
    instr_tail = instrs_[bci].get();
  }
  return instr_tail;
#else
  return instrs_[blk->end_ci() - 1].get();
#endif
}

void CFG::BuildCFG(const std::map<int, Block *> &labels) {
  // link
  for (size_t i = 0; i < bb_pool_.size(); ++i) {
    Block *bb = bb_pool_[i].get();
    const Instr *instr_tail = GetBlockTail(bb);
    bool is_fall_through = !Opcode(instr_tail->op()).IsNotFall();
    if (is_fall_through) {
      MS_EXCEPTION_IF_CHECK_FAIL(i + 1 < bb_pool_.size(), "Error byte code end");
      bb->AddSuccBB(bb_pool_[i + 1].get());
    }
    if (instr_tail->extra_jump() != nullptr) {
      const auto &it_bb = labels.find(instr_tail->extra_jump()->bci());
      MS_EXCEPTION_IF_CHECK_FAIL(it_bb != labels.cend(), "Target BB is not found");
      bb->AddSuccBB(it_bb->second);
    }
  }
}

static bool VisitBlock(Block *blk, std::vector<bool> *reach, std::vector<bool> *mark,
                       std::vector<Block *> *loop_heads) {
  if (reach->operator[](blk->id())) {
    if (mark->operator[](blk->id()) && !blk->is_loop_head()) {
      blk->set_is_loop_head(true);
      blk->set_is_loop_body(true);
      loop_heads->emplace_back(blk);
    }
    return blk->is_loop_body();
  }
  bool loop_body = false;

  blk->set_is_dead(false);
  reach->operator[](blk->id()) = true;
  mark->operator[](blk->id()) = true;
  auto iter = blk->succ_bbs().begin();
  for (; iter != blk->succ_bbs().end(); ++iter) {
    loop_body |= VisitBlock(*iter, reach, mark, loop_heads);
  }
  // If the current basic block (BB) is part of the loop body but not the loop header, and among the successor BBs of
  // the current BB there exists a BB with no successors, then that BB can also be considered part of the loop body.
  if (loop_body && !blk->is_loop_head()) {
    iter = blk->succ_bbs().begin();
    for (; iter != blk->succ_bbs().end(); ++iter) {
      if ((*iter)->succ_bbs().empty()) {
        (*iter)->set_is_loop_body(loop_body);
      }
    }
  }
  mark->operator[](blk->id()) = false;
  if (blk->is_loop_head()) {
    loop_heads->pop_back();
    return !loop_heads->empty();
  }
  if (!loop_heads->empty()) {
    blk->set_loop_head(loop_heads->back());
  }
  blk->set_is_loop_body(loop_body);
  return loop_body;
}

void CFG::MarkDeadBB() {
  if (bb_pool_.empty()) {
    return;
  }
  std::vector<bool> reach(bb_pool_.size());
  std::vector<bool> mark(bb_pool_.size());
  std::vector<Block *> loop_heads;
  VisitBlock(bb_pool_[0].get(), &reach, &mark, &loop_heads);
  for (const auto &i : bb_pool_) {
    if (reach[i->id()]) {
      continue;
    }
    i->set_is_dead(true);
  }
}

Block *CFG::GetBlockByBci(int bci) const {
  auto iter = std::find_if(bb_pool().begin(), bb_pool().end(), [bci](const std::unique_ptr<Block> &i) {
    return i->begin_ci() <= bci && bci < i->end_ci();
  });
  if (iter == bb_pool().end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "can't find block at " << bci;
  }
  return iter->get();
}

std::string CFG::ToString() const {
  std::ostringstream os;
  os << "*** Dump BB on [" << co_.ToString() << "] ***" << std::endl;
  for (const auto &bb : bb_pool_) {
    os << bb->ToString() << std::endl;
    for (int i = bb->begin_ci(), size = bb->end_ci(); i < size; ++i) {
      if (instrs_[i] != nullptr && instrs_[i]->op() != 0) {
        os << "  " << instrs_[i]->ToString() << std::endl;
      }
    }
  }
  return os.str();
}

CFG::BBIterator &CFG::BBIterator::operator++() {
  if (q_.empty()) {
    return *this;
  }
  Block *bb = q_.back();
  q_.pop_back();
  for (Block *bb_next : bb->succ_bbs()) {
    if (visit_[bb_next->id()]) {
      continue;
    }
    q_.push_back(bb_next);
    visit_[bb_next->id()] = true;
  }
  return *this;
}

const Liveness *CFG::GetLiveness() {
  if (liveness_ == nullptr) {
    liveness_ = std::make_unique<Liveness>(this);
    liveness_->Init();
  }
  return liveness_.get();
}

}  // namespace pijit
}  // namespace mindspore
