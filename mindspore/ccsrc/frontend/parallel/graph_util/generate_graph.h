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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/op_def.h"
#include "utils/hash_map.h"
#include "frontend/optimizer/opt.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
const char USING_HASH_NAME[] = "USING_HASH_NAME";
std::pair<bool, size_t> CheckAndGetValidIdxByOpDef(const ops::OpDefPtr &op_def, const std::string &op_name,
                                                   const std::string &attr_name, size_t limit_size);
// Get the operator's path where the operator has be defined
const char *GetOpPythonPath(const char *op_name);

// Init python operator Instance
ValuePtr CreateOpInstance(const OperatorAttrs &attrs, const OperatorName &op_name, const std::string &instance_name);
std::vector<AnfNodePtr> ConvertToRealInputs(const OperatorName &op_name, const std::string &instance_name,
                                            const AnfNodePtrList &inputs, const OperatorAttrs &attrs);
CNodePtr CreateCNodeByInputsAndAttr(const FuncGraphPtr &func_graph, const OperatorName &op_name,
                                    const std::string &instance_name, const AnfNodePtrList &inputs,
                                    const OperatorAttrs &attrs);
CNodePtr CreateNewCNodeForReplace(const CNodePtr &origin_node, const PrimitivePtr &new_prim);

AnfNodePtr CreateTypeInt(int64_t nbits);
AnfNodePtr CreateTypeFloat(int64_t nbits);
AnfNodePtr CreatInt64Imm(int64_t value);
AnfNodePtr CreateFP32Imm(float value);
AnfNodePtr CreateBoolImm(bool value);
AnfNodePtr CreateInt32Tensor(int64_t value, bool int64_type = false);
AnfNodePtr CreateFP32Tensor(float value);
AnfNodePtr CreateStringImm(std::string value);
AnfNodePtr ValuePtrToAnfNodePtr(const ValuePtr &value_ptr);
AnfNodePtr CreateTuple(const std::vector<int64_t> &tuple);
std::string HashInstanceName(const std::string &name);
void InsertVirtualPipelineEndNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager, size_t index,
                                  std::string end_flag = "pipeline_end");
CNodePtr CreateVirtualConverterBeginNode(const AnfNodePtr &input_cnode, size_t output_nums);
CNodePtr CreateVirtualConverterEndNode(const FuncGraphPtr &graph, const std::vector<CNodePtr> &input_cnodes);

class GenerateGraph {
 public:
  explicit GenerateGraph(const mindspore::HashMap<std::string, ValuePtr> &origin_attrs)
      : name_idx_(0), origin_attrs_(origin_attrs) {}
  Status Init(const CNodePtr &cnode);
  ~GenerateGraph() = default;
  AnfNodePtr virtual_input_node() { return virtual_input_node_; }
  AnfNodePtr NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs, const OperatorAttrs &prim_attrs);
  AnfNodePtr NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs);
  AnfNodePtr NewOpInst(const OperatorName &op_name);
  AnfNodePtr PushBack(const std::vector<AnfNodePtr> &inputs);

 private:
  CNodePtr cnode_;
  FuncGraphManagerPtr manager_;
  ScopePtr scope_;
  FuncGraphPtr func_graph_;
  AnfNodePtr virtual_input_node_;
  std::string instance_name_base_;
  int64_t name_idx_;
  mindspore::HashMap<std::string, ValuePtr> origin_attrs_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
