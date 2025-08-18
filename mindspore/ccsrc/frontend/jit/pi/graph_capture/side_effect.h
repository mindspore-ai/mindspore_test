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

#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include "frontend/jit/pi/graph_capture/node.h"

namespace mindspore {
namespace pijit {

class CodeGenerator;

// an unique data in the whole compilation
class SideEffectData {
 public:
  struct AttrCache {
    // a map of the modified object and it's modified attrs
    using AttrMap = std::map<std::string, ValueNode *>;
    std::map<ValueNode *, AttrMap> modified_attrs_;
  };

  struct GlobalCache {
    // a map of module and modified global dict
    using NameMap = std::map<std::string, ValueNode *>;
    std::map<std::string, NameMap> modified_globals_;
  };

  const auto &attr_cache() const { return attr_cache_; }
  const auto &global_cache() const { return global_cache_; }
  const auto &id_map() const { return id_map_; }
  const auto &modified_and_replaced_map() const { return modified_and_replaced_map_; }

  // track object and nodes
  void Track(PyObject *ptr, ValueNode *node) { (ptr ? (void)id_map_[ptr].insert(node) : (void)0); }
  void UnTrack(PyObject *ptr, ValueNode *node) { (ptr ? (void)id_map_[ptr].erase(node) : (void)0); }

  /**
   * record replaced node
   * NOTE: avoid this case:
   *    old_node.assign(new_node)
   *    old_node.assign(other)
   *    new_node.assign(other)
   * if replace 'old_node' by 'new_node', how to identify old_node.assign and new_node.assign ?
   * old_node must be unreached after replace and record. new_node is a new temporary node
   */
  void RecordModifiedAndReplacedNode(ValueNode *src_node, ValueNode *new_node);

  // merge attr modify operations
  void AddAttrData(const std::string &name, ValueNode *src, ValueNode *new_attr);

  // merge global modify operations
  void AddGlobalData(const std::string &module_name, const std::string &name, ValueNode *value);

  void ClearCache();

 private:
  // an unique map that record python object and nodes in the whole compilation
  // used to resolve object consistency
  std::map<PyObject *, std::set<ValueNode *>> id_map_;

  // an unique map of new value(key) and old_value(value)
  std::map<ValueNode *, ValueNode *> modified_and_replaced_map_;

  // optimization cache, record modified object
  // if record is reset, clean cache
  AttrCache attr_cache_;
  GlobalCache global_cache_;
};

class SideEffect {
 public:
  enum Type {
    kDefault,
    kSetGlobal,
    kBuiltinFunction,
    kBuiltinMethod,
    kTensorOptMethod,  // optimize side effect restore. Now only for tensor, extend this flag later
  };

  struct CacheResult {
    ValueNode *cache_value_;
    bool is_deleted_value_;
  };

  // find attribute from id_map and attr cache
  CacheResult LoadAttr(ValueNode *src, const std::string &name) const;

  // find global from global cache
  CacheResult LoadGlobal(const std::string &module_name, const std::string &name) const;

 public:
  SideEffect() = default;

  const auto &data() const { return data_; }
  const auto &nodes() const { return nodes_; }
  void set_data(const std::shared_ptr<SideEffectData> &data) { data_ = data; }

  // check the node is a side-effect record
  bool IsRecord(ValueNode *node) const { return nodes_.empty() ? false : nodes_.find(node) != nodes_.end(); }

  // check record is empty
  bool IsEmpty() const { return nodes_.empty(); }

  // return false if unsupported the side-effect
  bool Record(ValueNode *side_effect_node, Type type = Type::kDefault, std::string name = "");

  // return the original node(source, oldest version) if it's replaced, else return the node
  ValueNode *GetSource(ValueNode *node) const;

  // return the side-effect handler required nodes
  const std::set<ValueNode *> &GetRequiredNodes() const { return keep_alive_; }

  // The argument `node` is a side-effect node, and the function returns the required nodes used to restore
  // this side-effect operation.
  std::vector<ValueNode *> GetKeepAlive(ValueNode *node) const;

 private:
  struct Entry {
    ValueNode *node_;
    Type type_;
    size_t order_;
    std::string method_name_;
  };
  // add nodes to required
  void AddKeepAlive(const std::vector<ValueNode *> &inputs) { keep_alive_.insert(inputs.begin(), inputs.end()); }

  // get required node of the side-effect node
  std::vector<ValueNode *> GetKeepAlive(const Entry &) const;

  // if side-effect is function call, check it's supported
  bool CheckCallRecord(ValueNode *node, Type type, const std::string &name);

  // shared from other side-effect recorder
  std::shared_ptr<SideEffectData> data_;

  // record operations, check side-effect order
  std::map<ValueNode *, Entry> nodes_;

  // side-effect handler required nodes
  std::set<ValueNode *> keep_alive_;
};

// return the self node, if return nullptr, unsupported to handle side-effect
ValueNode *GetSelfFromKnownMethod(ValueNode *call_node, bool *is_method_descriptor = nullptr);

class SideEffectHandler {
 public:
  explicit SideEffectHandler(Graph *graph) : graph_(graph) {}
  virtual ~SideEffectHandler() = default;

  /// \brief The processing entry of side effect.
  ///
  /// \note Includes variable scope analysis, side effect collection, side effect effect reproduction, etc.
  void Run();
  /// \brief Collect the inputs of side effect operations.
  ///
  /// \return The inputs of side effect operations.
  std::vector<ValueNode *> GetSideEffectInputs() const;
  /// \brief Get the side effect operations.
  ///
  /// \return The side effect operations.
  const std::vector<ValueNode *> &GetSideEffect() const { return side_effect_nodes_; }
  /// \brief The optimization of side effect nodes.
  ///
  /// \param[in] nodes All the side effect nodes.
  ///
  /// \return The optimized side effect nodes
  static std::vector<ValueNode *> OptimizeSideEffect(const std::vector<ValueNode *> &nodes);

 private:
  /// \brief Reset the running environment for analysis.
  void ResetRunningEnvironment();
  /// \brief Collect the inputs of the captured nodes.
  ///
  /// \return The inputs include the parameters, interpretation execution nodes before the graph entry.
  std::vector<ValueNode *> CollectCapturedInputs() const;
  /// \brief Collect the captured nodes.
  ///
  /// \return The captured nodes.
  ///
  /// \note The nodes executed in the graph, need to be analyzed.
  std::vector<ValueNode *> CollectCapturedNodes() const;
  /// \brief The scope analysis of the call node.
  ///
  /// \param[in] node The call node that need to analyzed.
  void AnalyzeCallNodeScope(CallNode *node) const;
  /// \brief The scope analysis of the specified node.
  ///
  /// \param[in] node The specified node that need to analyzed.
  void AnalyzeNodeScope(ValueNode *node) const;
  /// \brief The scope analysis of the captured node.
  void ScopeAnalysis() const;
  /// \brief Group the captured nodes.
  ///
  /// \note Marked as a virtual machine node if There is no corresponding node in the graph.
  ///       Marked as a graph node if created only for graph.
  ///       Others marked as multi-purpose.
  void GroupCapturedNodes() const;
  /// \brief Collect the modified external variables.
  ///
  /// \return The modified external variables.
  std::vector<ValueNode *> CollectModifiedExternalVariables() const;
  /// \brief Collect the side effect operations.
  ///
  /// \return The side effect operations.
  ///
  /// \note Only side effects on external variables will be collected.
  std::vector<ValueNode *> CollectSideEffectOperations() const;
  /// \brief Initialize the version node map used to optimization.
  ///
  /// \param[in] vars All the modified external vars.
  void InitializeVersionNodeMaps(const std::vector<ValueNode *> &vars);
  /// \brief Revert the object that the call node is applied to the base version.
  ///
  /// \param[in] call_node The side effect node.
  void RebaseObjectVersion(CallNode *call_node) const;
  /// \brief Revert the object that the side effect is applied to the base version.
  ///
  /// \param[in] side_effect_nodes The side effect nodes will be handled.
  ///
  /// \return The side effect nodes whose object has been reverted.
  std::vector<ValueNode *> RebaseObjectVersionInSideEffects(const std::vector<ValueNode *> &side_effect_nodes) const;
  /// \brief Correct the variable if the module of the global not same as the graph.
  ///
  /// \param[in] nodes The side effect nodes will be handled.
  ///
  /// \return The side effect nodes have been handled
  std::vector<ValueNode *> CorrectVariableOfStoreGlobal(const std::vector<ValueNode *> &nodes) const;
  /// \brief Eliminating redundant side effects.
  ///
  /// \param[in] nodes All the side effect nodes.
  ///
  /// \return The side effect nodes have been handled
  static std::vector<ValueNode *> EliminateRedundantSideEffect(const std::vector<ValueNode *> &nodes);
  /// \brief Combining multiple side effects into one.
  ///
  /// \param[in] nodes All the side effect nodes.
  ///
  /// \return The side effect nodes have been handled
  ///
  /// \note scene 1 : Multiple Tensor setitem becomes one set data.
  ///       scene 2 : Multiple list store_subscr and all list item is set.
  std::vector<ValueNode *> MergeSideEffect(const std::vector<ValueNode *> &nodes) const;

  /// \brief The graph corresponding to the nodes being captured.
  Graph *graph_;
  /// \brief The graph corresponding to the nodes being captured.
  std::vector<ValueNode *> inputs_;
  /// \brief The nodes being captured.
  std::vector<ValueNode *> nodes_;
  /// \brief All the base versions of external variables and Corresponding instruction node.
  std::map<const AObject *, ValueNode *> ex_var_base_2_node_;
  /// \brief All the latest versions of external variables and Corresponding instruction node.
  std::map<const AObject *, ValueNode *> ex_var_latest_2_node_;
  /// \brief The side effect operations.
  std::vector<ValueNode *> side_effect_nodes_;
  /// \brief Indicate the status of side effect handler.
  ///        INT_MIN : Never Run, -1: Already Run without break point, > 0 : Already Run with break bci
  int break_bci_{INT_MIN};
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
