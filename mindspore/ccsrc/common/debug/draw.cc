/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include "ir/meta_func_graph.h"
#include "ir/primitive.h"
#include "ir/graph_utils.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/common.h"

namespace mindspore {
// namespace to support debug utils
namespace draw {
namespace {
// Only for ValueNode
std::string ValueType(const ValueNodePtr &node) {
  if (node == nullptr) {
    return "";
  }
  auto v = node->value();
  MS_EXCEPTION_IF_NULL(v);
  return v->type_name();
}
}  // namespace

// API of debug utils
void DrawNodes(const std::vector<AnfNodePtr> &nodes, OrderedMap<FuncGraphPtr, std::shared_ptr<BaseDigraph>> *sub_graphs,
               bool is_user) {
  if (sub_graphs == nullptr) {
    return;
  }
  for (auto &nd : nodes) {
    MS_EXCEPTION_IF_NULL(nd);
    auto sub_graph = nd->func_graph();
    if (sub_graph != nullptr) {
      auto gsub = (*sub_graphs)[sub_graph];
      if (gsub == nullptr) {
        if (is_user) {
          gsub = std::make_shared<ModelDigraph>(sub_graph->ToString());
        } else {
          gsub = std::make_shared<Digraph>(sub_graph->ToString());
        }
        (*sub_graphs)[sub_graph] = gsub;
      }
      if (!nd->isa<Parameter>()) {
        gsub->Node(nd, 0);
      }
    }
  }
}

void DrawValueNodes(const std::vector<AnfNodePtr> &nodes,
                    OrderedMap<FuncGraphPtr, std::shared_ptr<BaseDigraph>> *sub_graphs) {
  if (sub_graphs == nullptr) {
    return;
  }

  int dup_idx = 0;

  for (auto &nd : nodes) {
    for (auto &t : GetInputs(nd)) {
      MS_EXCEPTION_IF_NULL(t);
      MS_EXCEPTION_IF_NULL(nd);
      if (t->isa<ValueNode>() && (*sub_graphs).find(nd->func_graph()) != (*sub_graphs).end()) {
        (*sub_graphs)[nd->func_graph()]->Node(t, dup_idx);
        dup_idx++;
      } else if (t->isa<Parameter>() && (*sub_graphs).find(t->func_graph()) != (*sub_graphs).end()) {
        (*sub_graphs)[t->func_graph()]->Node(t, dup_idx);
        dup_idx++;
      }
    }
  }
}

void DrawEdges(const std::vector<AnfNodePtr> &nodes, const std::shared_ptr<BaseDigraph> &digraph, bool is_user) {
  if (digraph == nullptr) {
    return;
  }

  int dup_idx = 0;

  int offset = 0;
  if (is_user) {
    offset = 1;
  }

  // Draw edge
  for (auto &nd : nodes) {
    auto &succs = GetInputs(nd);
    auto num = succs.size();
    for (size_t i = 0; i < num; i++) {
      auto &t = succs.at(i);
      MS_EXCEPTION_IF_NULL(t);
      if (t->isa<ValueNode>() || t->isa<Parameter>()) {
        if ((!is_user) || (i != 0)) {
          // `SizeToInt(i) - offset` is just for printing as text
          digraph->Edge(t, nd, SizeToInt(i) - offset, dup_idx);
        }
        if (IsValueNode<FuncGraph>(t)) {
          auto const_graph = GetValueNode<FuncGraphPtr>(t);
          digraph->Edge(t, const_graph, dup_idx);
        }
        dup_idx++;
      } else {
        digraph->Edge(t, nd, SizeToInt(i) - offset, 0);
      }
    }
  }
}

void DrawByOpt(const std::string &filename, const FuncGraphPtr &func_graph, bool is_user) {
  if (func_graph == nullptr) {
    return;
  }
  auto ret = func_graph->get_return();
  auto nodes = DeepScopedGraphSearch(ret);

  std::shared_ptr<BaseDigraph> digraph;
  OrderedMap<FuncGraphPtr, std::shared_ptr<BaseDigraph>> sub_graphs;
  ChangeFileMode(filename, S_IWUSR);
  if (is_user) {
    digraph = std::make_shared<ModelDigraph>("mindspore", filename);
  } else {
    digraph = std::make_shared<Digraph>("mindspore", filename);
  }

  MS_EXCEPTION_IF_NULL(digraph);
  digraph->Start();

  // Draw nodes
  DrawNodes(nodes, &sub_graphs, is_user);

  // Draw ValueNodes on CNodes
  DrawValueNodes(nodes, &sub_graphs);

  // Draw subgraph
  for (const auto &gsub : sub_graphs) {
    digraph->SubGraph(gsub.first, gsub.second);
  }

  // Draw edge
  DrawEdges(nodes, digraph, is_user);

  digraph->End();
  // set file mode to read only by user
  ChangeFileMode(filename, S_IRUSR);
}

#ifdef ENABLE_DUMP_IR
void Draw(const std::string &filename, const FuncGraphPtr &func_graph) {
  const std::string dot_suffix = ".dot";
  const std::string filename_with_suffix =
    (filename.rfind(dot_suffix) != (filename.size() - dot_suffix.size())) ? (filename + dot_suffix) : filename;
  const std::string filepath = GetSaveGraphsPathName(Common::AddId(filename_with_suffix, dot_suffix));
  auto real_filepath = Common::CreatePrefixPath(filepath);
  if (!real_filepath.has_value()) {
    MS_LOG(ERROR) << "The export ir path: " << filepath << " is illegal.";
    return;
  }
  DrawByOpt(real_filepath.value(), func_graph, false);
}

void DrawUserFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph) {
  const std::string dot_suffix = ".dot";
  const std::string filepath = GetSaveGraphsPathName(Common::AddId(filename, dot_suffix));
  auto real_filepath = Common::CreatePrefixPath(filepath);
  if (!real_filepath.has_value()) {
    MS_LOG(ERROR) << "The export ir path: " << filepath << " is illegal.";
    return;
  }
  DrawByOpt(real_filepath.value(), func_graph, true);
}
#else
void Draw(const std::string &, const FuncGraphPtr &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in graphviz dot format is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}

void DrawUserFuncGraph(const std::string &, const FuncGraphPtr &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in graphviz dot format is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif

std::string Graphviz::Shape(const AnfNodePtr &node) {
  if (node == nullptr) {
    return "";
  }

  if (node->isa<CNode>()) {
    return "plaintext";
  }

  if (node->isa<Parameter>()) {
    return "octagon";
  }

  if (IsValueNode<FuncGraph>(node)) {
    return "oval";
  }

  return "plaintext";
}

std::string Graphviz::Color(const AnfNodePtr &node) const {
  if (node == nullptr) {
    return "";
  }

  if (node->isa<CNode>()) {
    return "cornsilk";
  }

  if (node->isa<Parameter>()) {
    return "paleturquoise";
  }

  if (IsValueNode<FuncGraph>(node)) {
    return "palegreen";
  }

  return "lavender";
}

void BaseDigraph::Start() {
  buffer_ << "digraph " << name_ << " {" << std::endl;
  buffer_ << "compound=true" << std::endl;
}

void BaseDigraph::Head(const AnfNodePtr &node, int id) {
  if (node == nullptr) {
    return;
  }

  buffer_ << "node" << node << "_" << id;
  if (node->isa<CNode>() || (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node))) {
    buffer_ << ":core";
  }
}

void BaseDigraph::Tail(const AnfNodePtr &node, int idx, int id) {
  if (node == nullptr) {
    return;
  }

  buffer_ << "node" << node << "_" << id;
  buffer_ << ":" << idx;
}

void BaseDigraph::Tail(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }
  buffer_ << "node" << func_graph->get_return() << "_" << 0;
}

void BaseDigraph::Edge(const AnfNodePtr &start, const FuncGraphPtr &end, int id_start) {
  Head(start, id_start);
  buffer_ << "->";
  Tail(end);

  buffer_ << "[lhead=cluster_" << end;
  buffer_ << ",dir=both,arrowhead=dot,style=filled,color=blue]";
  buffer_ << std::endl;
}

void BaseDigraph::End() {
  buffer_ << "}" << std::endl;

  if (fout_.is_open()) {
    fout_ << buffer_.str();
  }
}

void BaseDigraph::FuncGraphParameters(const FuncGraphPtr &key) {
  buffer_ << "parameters_" << key << "[shape=plaintext ";
  buffer_ << "label=<<table bgcolor='paleturquoise' cellspacing='0' cellborder='1' border='0'>";
  buffer_ << "<tr><td>parameters</td></tr>";
  int count = 0;
  for (auto &parameter : key->parameters()) {
    MS_EXCEPTION_IF_NULL(parameter);
    buffer_ << "<tr><td>";
    buffer_ << parameter->ToString();
    auto param = parameter->cast<ParameterPtr>();
    if (param && param->has_default()) {
      auto tensor_v = param->default_param();
      if (tensor_v && tensor_v->isa<tensor::Tensor>()) {
        auto tensor = tensor_v->cast<tensor::TensorPtr>();
        auto &shape = tensor->shape();
        std::ostringstream shape_str;
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(shape_str, ","));
        buffer_ << "[" << shape_str.str() << "]";
      }
    }
    buffer_ << "</td></tr>";
    count++;
    // Wrap the text.
    if (count % 10 == 0) {
      buffer_ << "\n";
    }
  }
  buffer_ << "</table>>,];";
}

void BaseDigraph::SubGraph(const FuncGraphPtr &key, const std::shared_ptr<BaseDigraph> &gsub) {
  if (key == nullptr || gsub == nullptr) {
    return;
  }

  std::string label;
  if (key->debug_info() != nullptr) {
    label = key->debug_info()->debug_name();
  }
  if (label.empty()) {
    label = gsub->name();
  }

  std::string label_managed = "[managed]";
  if (key->manager() == nullptr) {
    label_managed = "[not managed]";
  }
  label += label_managed;

  gsub->FuncGraphParameters(key);
  buffer_ << "subgraph cluster_" << key << "{" << std::endl;
  buffer_ << "id=cluster_" << key << std::endl;
  buffer_ << "label=\"" << label << "\"" << std::endl;
  buffer_ << "fontname=\"Courier New\"" << std::endl;
  buffer_ << gsub->buffer().str();
  buffer_ << "}" << std::endl;
}

Digraph::~Digraph() {
  try {
    if (fout_.is_open()) {
      fout_.close();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when closing file " << filename_;
  }
}

static std::string ReplaceAll(std::string str, const std::string &from, const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    (void)str.replace(start_pos, from.length(), to);
    // Handles case where 'to' is a substring of 'from'
    start_pos += to.length();
  }
  return str;
}

static void DrawValueNode(Graphviz *const graph_obj, const ValueNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph_obj);
  graph_obj->buffer() << "label=<<table port='core' cellborder='0' cellspacing='2' bgcolor='" << graph_obj->Color(node)
                      << "'>";
  graph_obj->buffer() << "<tr><td bgcolor='white'>" << ValueType(node) << "</td></tr>"
                      << "<tr><td>";
  if (std::string value_node_str = AnfDumpHandler::ValueNodeStr(node); !value_node_str.empty()) {
    graph_obj->buffer() << value_node_str;
  } else {
    std::ostringstream ss;
    MS_EXCEPTION_IF_NULL(node);
    ss << node->value()->ToString();
    std::string s = ReplaceAll(ss.str(), ", ", "<br/>");
    graph_obj->buffer() << s;
    ValuePtr value = node->value();
    if (value->isa<Primitive>()) {
      PrimitivePtr primitive = value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(primitive);
      graph_obj->buffer() << "</td></tr>"
                          << "<tr><td align='left'>";
      if (!primitive->instance_name().empty()) {
        graph_obj->buffer() << "instance name:"
                            << " " << primitive->instance_name() << "<br/>";
      }
      auto attrs = primitive->attrs();
      if (attrs.size() > 0) {
        graph_obj->buffer() << "</td></tr>"
                            << "<tr><td align='left'>";
        int i = 0;
        for (const auto &attr : attrs) {
          if (i != 0) {
            graph_obj->buffer() << "<br/>";
          }
          graph_obj->buffer() << attr.first << " ";
          if (attr.second == nullptr) {
            graph_obj->buffer() << " ";
          } else {
            graph_obj->buffer() << attr.second->ToString();
          }
          i++;
        }
      }
    }
  }
  graph_obj->buffer() << "</td></tr>"
                      << "</table>>,";
}

static void DrawParallelInfo(Graphviz *const graph_obj, const CNodePtr &node) {
  if (graph_obj == nullptr || node == nullptr) {
    return;
  }

  auto in_value = AnfDumpHandler::InStrategyValue(node);
  auto in_stage_value = AnfDumpHandler::InStrategyStageValue(node);
  if (in_value != nullptr && in_stage_value != nullptr) {
    auto num = node->size();
    graph_obj->buffer() << "<tr><td colspan='" << num << "' ";
    graph_obj->buffer() << "bgcolor='" << graph_obj->Color(node) << "'>";
    ValueTuplePtr strategy_tuple = std::make_shared<ValueTuple>(std::vector<ValuePtr>{in_stage_value, in_value});
    graph_obj->buffer() << "Strategy " << strategy_tuple->ToString();
    graph_obj->buffer() << "</td></tr>" << std::endl;
  }
}

static void DrawCNode(Graphviz *const graph_obj, const CNodePtr &node) {
  if (graph_obj == nullptr || node == nullptr || node->size() == 0) {
    return;
  }
  auto num = node->size();
  bool is_modelgraph = false;
  if (typeid(*graph_obj) == typeid(ModelDigraph)) {
    is_modelgraph = true;
    num -= 1;
  }

  graph_obj->buffer() << "label=<<table port='core'>" << std::endl;
  // Draw ports for CNode
  if (num > 0) {
    graph_obj->buffer() << "<tr>";
    for (size_t i = 0; i < num; i++) {
      graph_obj->buffer() << "<td port='" << i << "'>" << i << "</td>";
    }
    graph_obj->buffer() << "</tr>" << std::endl;
  }

  // Draw op name
  graph_obj->buffer() << "<tr><td";
  if (num > 0) {
    graph_obj->buffer() << " colspan='" << num << "'";
  }
  graph_obj->buffer() << " bgcolor='" << graph_obj->Color(node) << "'>";

  if (IsValueNode<Primitive>(node->input(0)) && is_modelgraph) {
    auto primitive = GetValueNode<PrimitivePtr>(node->input(0));
    graph_obj->buffer() << ReplaceAll(primitive->ToString(), ", ", "<br/>");
    auto attrs = primitive->attrs();
    if (attrs.size() > 0) {
      graph_obj->buffer() << "</td></tr>" << std::endl << "<tr><td";
      // Draw attrs
      if (num > 0) {
        graph_obj->buffer() << " colspan='" << num << "'";
      }
      graph_obj->buffer() << ">";
      int i = 0;
      for (auto &attr : attrs) {
        if (i != 0) {
          graph_obj->buffer() << "<br/>";
        }
        graph_obj->buffer() << attr.first << " " << attr.second->ToString();
        i++;
      }
    }
    graph_obj->buffer() << "CNode";
  } else {
    graph_obj->buffer() << "CNode(" << node->ToString() << ")";
  }

  graph_obj->buffer() << "</td></tr>" << std::endl;
  DrawParallelInfo(graph_obj, node);
  graph_obj->buffer() << "</table>>,";
}

void Digraph::Node(const AnfNodePtr &node, int id) {
  if (node == nullptr) {
    return;
  }

  buffer_ << "node" << node << "_" << id;
  buffer_ << "[";

  // Set fontname
  buffer_ << "fontname=\"Courier New\",";
  // Set label and shape
  buffer_ << "shape=" << Shape(node) << ",";
  if (node->isa<CNode>()) {
    DrawCNode(this, node->cast<CNodePtr>());
  } else if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    DrawValueNode(this, node->cast<ValueNodePtr>());
  } else {
    buffer_ << "label=\"" << node->ToString();
    if (IsValueNode<FuncGraph>(node)) {
      FuncGraphPtr next_net = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(next_net->debug_info());
      std::string next_net_name;
      if (next_net->debug_info() != nullptr) {
        next_net_name = next_net->debug_info()->debug_name();
      }
      if (!next_net_name.empty()) {
        buffer_ << "[" << next_net_name << "]";
      }
    }
    buffer_ << "\","
            << "style=filled,fillcolor=" << Color(node) << ",";
  }

  // Set URL for func graph
  if (IsValueNode<FuncGraph>(node)) {
    buffer_ << "URL=\"#cluster_" << GetValueNode(node) << "\",";
  }

  buffer_ << "]" << std::endl;
}

void Digraph::Edge(const AnfNodePtr &start, const AnfNodePtr &end, int idx, int id_start) {
  if (start == nullptr || end == nullptr) {
    return;
  }

  Head(start, id_start);
  buffer_ << "->";
  Tail(end, idx);

  buffer_ << "[arrowhead=vee,";

  // Check how many inputs for end
  if (end->isa<CNode>()) {
    auto cnode = end->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto num = cnode->size();
    if (idx == 0 && num > 1) {
      buffer_ << "style=dashed";
    }
  }
  buffer_ << "]" << std::endl;
}

ModelDigraph::~ModelDigraph() {
  try {
    if (fout_.is_open()) {
      fout_.close();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "exception when closing file " << filename_;
  }
}

std::string ModelDigraph::Shape(const AnfNodePtr &node) {
  if (node == nullptr) {
    return "";
  }

  if (node->isa<CNode>()) {
    return "plaintext";
  }

  if (node->isa<Parameter>()) {
    return "ellipse";
  }

  if (IsValueNode<FuncGraph>(node)) {
    return "oval";
  }

  return "plaintext";
}

void ModelDigraph::Node(const AnfNodePtr &node, int id) {
  if (node == nullptr) {
    return;
  }

  if (IsValueNode<Primitive>(node)) {
    return;
  }

  buffer_ << "node" << node << "_" << id;
  buffer_ << "[";

  // Set fontname
  buffer_ << "fontname=\"Courier New\",";
  // Set label and shape
  buffer_ << "shape=" << Shape(node) << ",";
  if (node->isa<CNode>()) {
    DrawCNode(this, node->cast<CNodePtr>());
  } else if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    DrawValueNode(this, node->cast<ValueNodePtr>());
  } else {
    buffer_ << "label=\"" << node->ToString() << "\",";
    buffer_ << "style=filled,fillcolor=" << Color(node) << ",";
  }

  // Set URL for func graph
  if (IsValueNode<FuncGraph>(node)) {
    buffer_ << "URL=\"#cluster_" << GetValueNode(node) << "\",";
  }

  buffer_ << "]" << std::endl;
}

void ModelDigraph::Edge(const AnfNodePtr &start, const AnfNodePtr &end, int idx, int id_start) {
  if (start == nullptr || end == nullptr) {
    return;
  }

  Head(start, id_start);
  buffer_ << "->";
  Tail(end, idx);

  buffer_ << "[arrowhead=vee,";
  buffer_ << "]" << std::endl;
}
}  // namespace draw
}  // namespace mindspore
