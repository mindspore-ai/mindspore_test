/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/free_variables_eliminate.h"

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/action.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {

FuncGraphPtr PartialEliminateOptPass(const pipeline::ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(resource);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig partial_eliminate_opt_ = opt::OptPassConfig(
    {irpass.partial_eliminate_, irpass.switch_partial_eliminater_, irpass.switch_layer_partial_eliminater_});
  opt::OptPassGroupMap map({{"partial_eliminate_", partial_eliminate_opt_}});

  auto after_lift_opt = opt::Optimizer::MakeOptimizer("partial_eliminate", resource, map);

  FuncGraphPtr opt_fg = nullptr;
  ProfileExecute(MsProfile::GetProfile()->Step("partial_eliminate_before_grad"),
                 [&after_lift_opt, func_graph, &opt_fg]() { opt_fg = after_lift_opt->step(func_graph, true); });
  return opt_fg;
}

FuncGraphVector PartialEliminateMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  FuncGraphVector opt_fgs;
  for (const auto &func_graph : func_graphs) {
    auto opt_fg = PartialEliminateOptPass(new_res, func_graph);
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
    }
#endif
    opt_fgs.push_back(opt_fg);
  }
  return opt_fgs;
}

bool CheckExistFv(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(nodes.begin(), nodes.end(), [&func_graph](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    return (node->func_graph() != func_graph);
  });
}

std::string GetRefKeyFromAbstractRef(const abstract::AbstractRefPtr &abs_ref) {
  MS_EXCEPTION_IF_NULL(abs_ref);
  MS_EXCEPTION_IF_NULL(abs_ref->ref_key_value());
  auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
  if (ref_key == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract is wrong: " << abs_ref->ToString();
  }
  return ref_key->value();
}

std::map<std::string, AnfNodePtr> GetParameterMap(const std::vector<AnfNodePtr> &params) {
  std::map<std::string, AnfNodePtr> params_map;
  for (const auto &param : params) {
    auto param_abs = param->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);
    auto abs_ref = param_abs->cast<abstract::AbstractRefPtr>();
    if (abs_ref == nullptr) {
      continue;
    }
    auto ref_key_str = GetRefKeyFromAbstractRef(abs_ref);
    params_map[ref_key_str] = param;
  }
  return params_map;
}

void MergeParameters(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  auto manager = optimizer->manager();
  const auto &fg_used_total = func_graph->func_graphs_used_total();
  for (const auto &fg : fg_used_total) {
    const auto &inner_params = fg->parameters();
    std::map<std::string, AnfNodePtr> ref_key_nodes = GetParameterMap(inner_params);
    const auto &nodes = TopoSort(fg->get_return());
    for (const auto &node : nodes) {
      if (!node->isa<Parameter>()) {
        continue;
      }
      auto abs = node->abstract();
      if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
        continue;
      }
      auto ref_abs = abs->cast<abstract::AbstractRefPtr>();
      const auto &ref_key_str = GetRefKeyFromAbstractRef(ref_abs);
      const auto &iter = ref_key_nodes.find(ref_key_str);
      if (iter == ref_key_nodes.end()) {
        continue;
      }
      const auto &real_param = iter->second;
      if (real_param != node) {
        manager->Replace(node, real_param);
      }
    }
  }
}
}  // namespace

FuncGraphPtr LiftFv(const pipeline::ResourceBasePtr &resource, const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  FuncGraphPtr new_fg = LiftingClone(func_graph);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
  }
#endif
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, func_graph->return_node())
      << "Parameter resources is not a pipeline::Resource";
  }
  auto opt_fg = PartialEliminateOptPass(new_res, new_fg);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
  }
#endif
  return opt_fg;
}

FuncGraphVector LiftFvMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    for (const auto &func_graph : func_graphs) {
      DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
    }
  }
#endif
  bool has_used_fg = std::any_of(func_graphs.cbegin(), func_graphs.cend(), [](const FuncGraphPtr &func_graph) {
    return func_graph->func_graphs_used().size() != 0;
  });
  // All func_graphs being graded don't have used funcgraphs, no need to do lifting clone.
  if (!has_used_fg) {
    return func_graphs;
  }
  FuncGraphVector new_fgs = LiftingCloneMulti(func_graphs);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    for (const auto &new_fg : new_fgs) {
      DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
    }
  }
#endif
  return PartialEliminateMulti(resource, new_fgs);
}

FuncGraphPtr FreeVariablesEliminate(FuncGraphPtr *func, const opt::OptimizerPtr &optimizer) {
  FuncGraphPtr func_graph = *func;
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);

  bool exist_fv = CheckExistFv(func_graph);
  MS_LOG(DEBUG) << "Exist free variables: " << exist_fv;
  if (!exist_fv) {
    return func_graph;
  }
  parse::ClearCNodeAbstract(func_graph);
  abstract::AbstractBasePtrList new_args_spec;
  (void)std::transform(func_graph->parameters().begin(), func_graph->parameters().end(),
                       std::back_inserter(new_args_spec),
                       [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
  auto res = std::make_shared<pipeline::Resource>();
  MS_LOG(DEBUG) << "LiftingClone for func_graph: " << func_graph->ToString();
  auto new_func_graph = pipeline::Renormalize(res, func_graph, new_args_spec);
  const auto &resources = optimizer->resource();
  new_func_graph = LiftFv(resources, new_func_graph);
  MergeParameters(new_func_graph, optimizer);

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_free_variable_eliminate.ir", new_func_graph);
  }
#endif
  return new_func_graph;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
