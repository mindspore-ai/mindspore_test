/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
#define MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ir/value.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "pipeline/jit/pi/graph_capture/abstract_wrapper.h"

namespace mindspore {
namespace pijit {
class FuncGraphBuilder;
using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;
class AbstractWrapper;
using AbstractWrapperPtr = std::shared_ptr<AbstractWrapper>;

class FuncGraphBuilder {
 public:
  explicit FuncGraphBuilder(bool is_top = false) : graph_(std::make_shared<FuncGraph>()) {
    if (is_top) {
      parse::Parser::UpdateTopFuncGraph(graph_);
      mng_ = Manage(graph_, true);
      graph_->set_manager(mng_);
    }
  }
  virtual ~FuncGraphBuilder() { key_to_node_.clear(); }

  /// \brief Add an input parameter to the graph.
  ///
  /// \param[in] abstract_wrapper The key to find node in function graph builder.
  ///
  /// \return The AbstractWrapperPtr for subgraph input.
  AbstractWrapperPtr AddSubGraphInput(const AbstractWrapperPtr abstract_wrapper);

  FuncGraphManagerPtr manager() const { return mng_; }

  void set_manager(const FuncGraphManagerPtr &mng) {
    mng_ = mng;
    graph_->set_manager(mng_);
  }

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper  of the infer result.
  AbstractWrapperPtr AddNode(const py::object &callable_obj, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNode(const ValuePtr &callable_value, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionEx.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionEx(const py::object &callable_obj,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionEx.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionEx(const ValuePtr &callable_value,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionKw.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionKw(const py::object &callable_obj,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionKw.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionKw(const ValuePtr &callable_value,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a python object to graph.
  ///
  /// \param[in] object The python object add to graph.
  ///
  /// \return Indicate whether the python object add to graph successfully.
  AbstractWrapperPtr AddAttrPythonObject(const py::object &object);

  /// \brief Add a binary operation cnode to the graph.
  ///
  /// \param[in] opcode The binary operation code.
  /// \param[in] inputs_abstract_wrapper The abstract wrapper for inputs.
  ///
  /// \return The python object of the infer result.
  AbstractWrapperPtr AddMultiNode(const std::string &name, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add an output node to the graph.
  ///
  /// \param[in] output_obj The output python object.
  /// \param[in] is_top_graph Indicate whether the graph to add output is top graph.
  ///
  /// \return Return true if the output object can be used as the output of the graph.
  bool AddOutput(const AbstractWrapperPtr &abstract_wrapper, bool is_top_graph = true);

  /// \brief Clear all output node of the graph.
  void ClearOutputNodes() { output_nodes_.clear(); }

  size_t GetOutputSize() const { return output_nodes_.size(); }

  /// \brief Get the callable python primitive or function.
  ///
  /// \param[in] obj The method of a python object.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertMethod(const py::object &obj);

  /// \brief Get the callable python primitive, meta_func_graph or function.
  ///
  /// \param[in] obj The python object of a function.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertFunction(const py::object &obj);

  /// \brief Check if the python object is a function which can be constantly folded.
  ///
  /// \param[in] obj A python object.
  ///
  /// \return Return true if the python object is a function which can be constantly folded.
  static bool CanConstantFoldFunc(const py::object &obj);

  /// \brief Check if the python object is valid as the callable object in graph.
  ///
  /// \param[in] obj A python object.
  ///
  /// \return Return true if the python object is valid as the callable object in graph.
  static bool ValidateCallableObject(const py::object &obj);

  /// \brief Set the final outputs and get the graph.
  ///
  /// \param[in] force Allows getting the graph when the outputs have not yet been added.
  ///
  /// \return The graph constructed.
  FuncGraphPtr graph(bool force = false);

  /// \brief Clear abstract for nodes.
  void ClearNodeAbstract();

  /// \brief Set the name of the func_graph.
  ///
  /// \param[in] name The func_graph name to set.
  void SetGraphName(const std::string &name);

  void AddPrevBuilder(const FuncGraphBuilderPtr &builder);

  const std::vector<FuncGraphBuilder *> &prev_builders() const { return prev_builders_; }

  void UpdateNodesMap(const AbstractWrapperPtr &key, const AnfNodePtr &node) {
    (void)key_to_node_.insert_or_assign(key, node);
  }

  AnfNodePtr ReadLocalVariable(const AbstractWrapperPtr &abstract_wrapper);

  AbstractWrapperPtr AddLocalVariable(const py::object &obj);

  /// \brief Add a custom node to the graph.
  ///
  /// \param[in] wrapper The abstract wrapper corresponding to the node.
  /// \param[in] node The node will be added.
  ///
  /// \note Nodes created during the conversion of Dict nodes need to be added to the graph using this method.
  void AddLocalVariableNode(const AbstractWrapperPtr &wrapper, const AnfNodePtr &node);

  AbstractWrapperPtr BuildGradNetNode(const ValuePtr &callable_value, const py::object &callable_obj,
                                      const AbstractWrapperPtrList &inputs_abstract_wrapper);

  AbstractWrapperPtr BuildGradNode(const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                   const AbstractWrapperPtrList &inputs);

  static FuncGraphPtr BuildCallForwardGraphForGrad(const FuncGraphPtr &fg, const std::vector<size_t> &arg_len,
                                                   bool is_cell);

  AbstractWrapperPtr AddTopGraphArgInput(const py::object &object);

  AbstractWrapperPtr AddTopGraphVargsInputs(const py::object &vargs);

  AbstractWrapperPtr AddTopGraphKwargsInputs(const py::object &vargs);

  AnfNodePtr FindNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper);

  AnfNodePtr GetNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper);

  AbstractWrapperPtr AddAttributeInput(const py::object &object);

  size_t origin_top_input_num() const { return origin_top_input_num_; }

 private:
  AnfNodePtr ConvertObjToNode(const py::object &input_obj);
  AnfNodePtr ConvertParameterTupleToNode(const py::object &input_obj);
  AnfNodePtr ConvertPyTupleListToNode(const py::object &obj);
  AnfNodePtr ConvertPyDictToNode(const py::dict &dict);

  AbstractWrapperPtr AddNodeWithAbstract(const ValuePtr &value, const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                         const AbstractBasePtr &abstract);

  bool GetInputNodesAndAbstracts(const ValuePtr &callable_value, const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                 std::vector<AnfNodePtr> *input_node_list,
                                 std::vector<AbstractBasePtr> *input_abs_list);

  CNodePtr DoPrimitiveInferAndCheck(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                    const AbstractBasePtrList &args_abs_list);
  CNodePtr AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                             const AbstractBasePtrList &args_abs_list);

  AbstractWrapperPtr TryToAddNode(const ValuePtr &callable_value,
                                  const AbstractWrapperPtrList &inputs_abstract_wrapper);

  AbstractWrapperPtr HandleGrad(const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                const AbstractWrapperPtrList &inputs);

  AbstractBasePtr FetchFuncGraphOutputAbstract(const ValuePtr &value) const;

  void UpdateParameterFuncGraph(const AnfNodePtr &node);

  void MarkNodeIsolated(const AnfNodePtr &node, bool force);

  void EraseCandidateIsolatedNode(const AnfNodePtr &node);

  AnfNodePtr GenerateOutputNode();

  AnfNodePtr AttachIsolatedNode(const AnfNodePtr &node) const;

  FuncGraphPtr graph_{nullptr};
  bool has_set_output_{false};
  HashMap<AbstractWrapperPtr, AnfNodePtr> key_to_node_;
  std::vector<AnfNodePtr> output_nodes_;

  // Store all isolated nodes for graph which should be appended to the output of graph.
  std::vector<AnfNodePtr> isolated_nodes_;

  // Store all previous builders for subgraph call and control flow.
  std::vector<FuncGraphBuilder *> prev_builders_;

  FuncGraphManagerPtr mng_;
  size_t origin_top_input_num_ = 0;
};
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
