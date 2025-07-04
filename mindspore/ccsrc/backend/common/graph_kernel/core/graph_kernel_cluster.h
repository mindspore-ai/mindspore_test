/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_

#include <vector>
#include <memory>
#include <sstream>
#include <set>
#include <string>
#include <algorithm>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
// Binary Indexed Trees for max value
class BitMax {
 public:
  // cluster_id starts from 0, but value in binary indexed tree starts from 1.
  explicit BitMax(size_t n) : vec_(n + 1) { std::iota(vec_.begin(), vec_.end(), 0); }
  void SetMax(size_t i, size_t val) {
    i++;
    while (i < vec_.size()) {
      vec_[i] = std::max(vec_[i], val);
      i += LowBit(i);
    }
  }

  size_t FindMax(size_t i) {
    i++;
    if (i >= vec_.size()) {
      i = vec_.size() - 1;
    }
    size_t result = 0;
    while (i > 0) {
      result = std::max(result, vec_[i]);
      i -= LowBit(i);
    }
    return result;
  }

 private:
  size_t LowBit(size_t x) const { return x & (-x); }
  std::vector<size_t> vec_;
};
using BitMaxPtr = std::shared_ptr<BitMax>;

class Graph;
using GraphPtr = std::shared_ptr<Graph>;
class Graph {
  struct Cluster {
    size_t cluster_id_;       // node_id of the representative.
    size_t cluster_size_{1};  // size of cluster, composite node is considered as one node.
    size_t max_id_;
    std::set<size_t> inputs_;  // inputs' cluster_id.
    size_t seed_{0};           // visited flag of dfs.

    Cluster(size_t node_id, const AnfNodePtr &node, const mindspore::HashMap<AnfNodePtr, size_t> &node_idx_map);
    ~Cluster() = default;

    void Merge(Cluster *other_cluster);

    // clean the info to free memory.
    void Clean() {
      inputs_.clear();
      cluster_size_ = 0;
    }
  };  // struct Cluster

 public:
  static GraphPtr Build(const FuncGraphPtr &func_graph, bool aggressive_cut, AnfNodePtrList *nodes = nullptr,
                        HashMap<AnfNodePtr, size_t> *node_idx_map = nullptr);
  ~Graph() = default;

  // find the representative of the cluster
  size_t Find(size_t node_id);

  // merge clusters, the smallest cluster id will be the new cluster id.
  void Merge(const std::vector<size_t> &candidates);

  // Collect nodes together that are in the same cluster.
  std::vector<std::vector<size_t>> CollectClusters();

  using VisitFunc = std::function<IncludeType(size_t)>;
  void Dfs(size_t node_id, const VisitFunc &visitor);

  // Get cluster size
  size_t GetSize(size_t cluster_id) { return clusters_[Find(cluster_id)].cluster_size_; }

  size_t GetMaxIdWithCutStrategy(size_t cluster_id) {
    return (bitmax_ == nullptr) ? GetMaxId(cluster_id) : bitmax_->FindMax(Find(cluster_id));
  }
  // Get max id in cluster
  size_t GetMaxId(size_t cluster_id) { return clusters_[Find(cluster_id)].max_id_; }

  // Get cluster's inputs
  const std::set<size_t> &GetInputs(size_t cluster_id);

  bool HasCircle();

  // public constructor for std::make_shared, do not call it manually.
  Graph(const AnfNodePtrList &nodes, const HashMap<AnfNodePtr, size_t> &node_idx_map, bool aggressive_cut);

 private:
  void RefreshInputs(size_t i);
  void DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor);

  std::vector<Cluster> clusters_;
  BitMaxPtr bitmax_{nullptr};
  size_t seen_{0};
};  // Graph

class CircleChecker {
 public:
  explicit CircleChecker(const GraphPtr &graph) : graph_(graph) {}
  ~CircleChecker() = default;

  void RemoveCircle(std::vector<size_t> *candidates);

 private:
  bool CheckCircle(size_t basenode);

  // remove all circle nodes from candidates
  void RemoveCircleNodesFromCandidates();

  GraphPtr graph_;               // bind the global graph
  std::set<size_t> candidates_;  // bind the input candidates
  std::vector<size_t> circle_nodes_;
  std::set<size_t> acyclic_nodes_;
};  // CircleChecker

class GraphKernelCluster : public opt::Pass {
 public:
  explicit GraphKernelCluster(const std::string &pass_name = "graph_kernel_cluster") : Pass(pass_name) {}
  ~GraphKernelCluster() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  virtual std::vector<PrimitivePtr> GetClusterableOpList() { return {}; }
  virtual bool IsClusterableOp(const AnfNodePtr &node) = 0;
  void Init(const FuncGraphPtr &func_graph);
  void GraphMerge(const FuncGraphPtr &func_graph, bool aggressive_cut);
  bool Process(const FuncGraphPtr &func_graph);
  std::vector<size_t> FindCandidates(size_t basenode_id);
  void RemoveWildGetitem(std::vector<size_t> *candidates);
  virtual void CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id);
  void DumpClusterInfo(const AnfNodePtrList &old_nodes, const AnfNodePtr &new_node);
  void DumpToFile();
  void Clean() {
    nodes_.clear();
    graph_ = nullptr;
  }

  GraphPtr graph_{nullptr};
  std::vector<AnfNodePtr> nodes_;
  std::stringstream dump_buf_;
  std::vector<PrimitivePtr> op_list_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_
