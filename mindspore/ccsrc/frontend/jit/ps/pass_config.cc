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

#include "frontend/jit/ps/pass_config.h"
#include <sstream>
#include <memory>
#include "utils/file_utils.h"
#include "frontend/optimizer/auto_monad_eliminate.h"
#include "frontend/optimizer/cse_pass.h"
#include "frontend/optimizer/irpass/branch_culling.h"
#include "frontend/optimizer/irpass/expand_dump_flag.h"
#include "frontend/optimizer/irpass/meta_fg_eliminate.h"
#include "frontend/optimizer/irpass/parameter_eliminate.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"

namespace mindspore {
namespace opt {
bool FilterPass(const std::string &pass_key) { return PassConfigure::Instance().FilterPass(pass_key); }

void UpdateRunningPasses(const std::string &pass_key) { PassConfigure::Instance().UpdateRunningPasses(pass_key); }
namespace {
OptPassConfig GeneratePassConfig(const py::dict &pass) {
  const auto renormalize_ = py::str("renormalize");
  const auto once_ = py::str("once");
  const auto sensitive_ = py::str("sensitive");
  const auto opt_pass_func_ = py::str("pass_func");
  const auto substitutions_ = py::str("list");
  bool is_once = false;
  bool global_sensitive = false;
  if (pass.contains(renormalize_) && pass[renormalize_].cast<bool>()) {
    return OptPassConfig::Renormalize();
  }
  if (pass.contains(once_) && pass[once_].cast<bool>()) {
    is_once = true;
  }
  if (pass.contains(sensitive_) && pass[sensitive_].cast<bool>()) {
    global_sensitive = true;
  }

  if (pass.contains(opt_pass_func_)) {
    auto pass_func_name = pass[opt_pass_func_].cast<std::string>();
    auto pass_func = PassConfigure::Instance().GetOptimizeFunc(pass_func_name);
    if (pass_func) {
      return OptPassConfig(pass_func);
    }
    MS_LOG(ERROR) << "There is not pass function: " << pass_func_name;
    return OptPassConfig::Renormalize();
  }
  if (pass.contains(substitutions_)) {
    auto substitutions = pass[substitutions_];
    std::vector<SubstitutionPtr> substitution_list;
    for (auto it = substitutions.begin(); it != substitutions.end(); ++it) {
      auto substitution_name = it->cast<std::string>();
      auto substitution = PassConfigure::Instance().GetSubstitution(substitution_name);
      if (substitution != nullptr) {
        substitution_list.emplace_back(substitution);
      } else {
        MS_LOG(ERROR) << "There is not substitution: " << substitution_name;
      }
    }
    if (substitution_list.empty()) {
      MS_LOG(ERROR) << "The substitution is empty. ";
      return OptPassConfig::Renormalize();
    }
    return OptPassConfig(substitution_list, is_once, global_sensitive);
  }
  MS_LOG(ERROR) << "The OptPassConfig is error. " << pass;
  return OptPassConfig::Renormalize();
}

OptPassGroupMap GeneratePassGroup(const py::list &pass_group_map) {
  OptPassGroupMap pass_map;
  for (auto it = pass_group_map.begin(); it != pass_group_map.end(); ++it) {
    if (!py::isinstance<py::dict>(*it)) {
      MS_LOG(ERROR) << "The pass_group_map is error.  It is not a dict. " << pass_group_map;
      continue;
    }
    auto conf_dict = it->cast<py::dict>();
    const auto kName = py::str("name");
    std::string name = "pass_config";
    if (conf_dict.contains(kName)) {
      name = conf_dict[kName].cast<std::string>();
    }
    auto passCfg = GeneratePassConfig(conf_dict);
    pass_map.emplace_back(name, passCfg);
  }
  return pass_map;
}

std::shared_ptr<Optimizer> GenerateOptimizer(const py::dict &optimizeCfg) {
  const auto name_ = py::str("name");
  const auto once_ = py::str("once");
  const auto renormalize_ = py::str("renormalize");
  const auto node_first_ = py::str("node_first");
  const auto pass_group_ = py::str("pass_group");
  std::string name;
  pipeline::ResourceBasePtr resource;
  OptPassGroupMap passes;
  bool run_only_once = false;
  bool watch_renormalize = false;
  bool traverse_nodes_first = true;
  if (optimizeCfg.contains(renormalize_)) {
    watch_renormalize = optimizeCfg[renormalize_].cast<bool>();
  }
  if (optimizeCfg.contains(once_)) {
    run_only_once = optimizeCfg[once_].cast<bool>();
  }
  if (optimizeCfg.contains(node_first_)) {
    traverse_nodes_first = optimizeCfg[node_first_].cast<bool>();
  }
  if (optimizeCfg.contains(name_)) {
    name = optimizeCfg[name_].cast<std::string>();
  }
  if (optimizeCfg.contains(pass_group_)) {
    auto py_pass_map = optimizeCfg[pass_group_];
    if (!py::isinstance<py::list>(py_pass_map)) {
      MS_LOG(ERROR) << "The optimizeCfg is error. " << optimizeCfg;
      return nullptr;
    }
    auto pass_group = GeneratePassGroup(py_pass_map);
    return Optimizer::MakeOptimizer(name, resource, pass_group, run_only_once, watch_renormalize, traverse_nodes_first);
  }
  MS_LOG(ERROR) << "There is not pass_group: " << optimizeCfg;
  return nullptr;
}

void GeneratePassItemVector(const py::list &optimize_cfg, std::vector<PassConfigure::PassItem> *passes) {
  for (auto it = optimize_cfg.begin(); it != optimize_cfg.end(); ++it) {
    if (!py::isinstance<py::dict>(*it)) {
      MS_LOG(ERROR) << "The optimize_cfg is error. Item: " << *it;
      continue;
    }
    const auto &optimizerCfg = it->cast<py::dict>();
    const auto name_ = py::str("name");
    const auto function_ = py::str("fun");
    if (optimizerCfg.contains(function_)) {
      auto funcName = optimizerCfg[function_].cast<std::string>();
      auto passName = optimizerCfg[name_].cast<std::string>();
      auto func = PassConfigure::Instance().GetPassFunc(funcName);
      if (func) {
        passes->emplace_back(PassConfigure::PassItem(passName, func));
      } else {
        MS_LOG(ERROR) << "The optimize_cfg is error. There is not pass function: " << funcName;
      }
      continue;
    }
    auto optimizer = GenerateOptimizer(optimizerCfg);
    if (optimizer) {
      auto f = [optimizer](pipeline::ResourcePtr resource) { return (*optimizer)(resource); };
      passes->emplace_back(PassConfigure::PassItem(optimizer->name(), f));
    } else {
      MS_LOG(ERROR) << "The optimize_cfg is error. Item: " << *it;
    }
  }
}
}  // namespace

void PassConfigure::SetOptimizeConfig(const py::list &optimize_cfg) {
  irpass::OptimizeIRPassLib();
  irpass::ResolveIRPassLib();
  irpass::GradPartialPassLib();
  std::vector<PassConfigure::PassItem> pass_items;
  GeneratePassItemVector(optimize_cfg, &pass_items);
  SetPasses(pass_items);
}

PassConfigure &PassConfigure::Instance() {
  static PassConfigure instance;
  return instance;
}

std::string PassConfigure::GetOptimizeConfig() {
  std::stringstream out;
  out << std::endl << "Opt graph functions:" << std::endl;
  constexpr auto tab_sp = "\t";
  constexpr auto new_line_sp = "\r\n";
  constexpr int number_3 = 3;
  int i = 0;
  for (const auto &item : opt_func_map_) {
    out << item.first << (++i % number_3 != 0 ? tab_sp : new_line_sp);
  }
  out << std::endl << std::endl << "Pass functions:" << std::endl;
  i = 0;
  for (const auto &item : pass_func_map_) {
    out << item.first << (++i % number_3 != 0 ? tab_sp : new_line_sp);
  }
  out << std::endl << std::endl << "Opt substitutions:" << std::endl;
  i = 0;
  for (const auto &item : substitution_map_) {
    out << item.first << (++i % number_3 != 0 ? tab_sp : new_line_sp);
  }
  out << std::endl << std::endl << "Opt PassItems:" << std::endl;
  i = 0;
  for (const auto &item : passes_) {
    out << item.first << tab_sp << (++i % number_3 != 0 ? tab_sp : new_line_sp);
  }
  out << std::endl;
  return out.str();
}
py::list PassConfigure::GetRunningPasses() {
  py::list ret;
  for (const auto &item : running_passes_) {
    ret.append(py::str(item.c_str()));
  }
  return ret;
}

void PassConfigure::SetConfigPasses(const py::list &cfg_passes) {
  cfg_passes_.clear();
  running_passes_.clear();
  if (!cfg_passes.empty()) {
    for (auto l : cfg_passes) {
      if (!l.is_none()) {
        cfg_passes_.insert(py::str(l));
      }
    }
  }
}

inline void Strim(std::string *str) {
  MS_EXCEPTION_IF_NULL(str);
  auto &s = *str;
  auto end = std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); });
  s.erase(s.begin(), end);
  auto begin = std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
  s.erase(begin, s.end());
}

void SavePassesConfig(const std::string &func_graph) {
  auto running_passes = opt::PassConfigure::Instance().GetRunningPasses();
  MS_LOG(INFO) << "Running_passes: " << py::str(running_passes);

  // Clear custom config passes.
  py::list passes;
  opt::PassConfigure::Instance().SetConfigPasses(passes);
  auto path = common::GetCompileConfig("AUTO_PASSES_OPTIMIZE_PATH");
  Strim(&path);
  if (path.empty() || py::len(running_passes) < 1) {
    return;
  }
  path += "/" + func_graph + "_pass.conf";
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file ./" << func_graph << ".conf failed.";
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  auto ofs = FileUtils::OpenFile(realpath.value(), std::ios::out);
  if (ofs == nullptr) {
    MS_LOG(ERROR) << "Open the file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }
  for (auto &pass : running_passes) {
    *ofs << py::str(pass) << std::endl;
  }
  ofs->close();
  delete ofs;
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void LoadPassesConfig(const std::string &func_graph) {
  auto path = common::GetCompileConfig("AUTO_PASSES_OPTIMIZE_PATH");
  Strim(&path);
  MS_LOG(INFO) << "AUTO_PASSES_OPTIMIZE_PATH: " << path;
  if (path.empty()) {
    return;
  }

  std::string realpath = path + "/" + func_graph + "_pass.conf";
  auto ifs = FileUtils::OpenFile(realpath, std::ios::in);
  if (ifs == nullptr) {
    MS_LOG(INFO) << "Open the file '" << realpath << "' failed!" << ErrnoToString(errno);
    return;
  }
  constexpr int kNumber256 = 256;
  char buffer[kNumber256];
  py::list cfg_passes;
  ifs->getline(buffer, kNumber256);
  while (ifs->good()) {
    cfg_passes.append(py::str(buffer));
    ifs->getline(buffer, kNumber256);
  }
  ifs->close();
  delete ifs;
  opt::PassConfigure::Instance().SetConfigPasses(cfg_passes);
  MS_LOG(INFO) << "Set Config passes: " << py::str(cfg_passes);
}

namespace irpass {
REGISTER_OPT_PASS_CLASS(ParameterEliminator)
REGISTER_OPT_PASS_CLASS(AutoMonadEliminator)
REGISTER_OPT_PASS_CLASS(CSEPass)
REGISTER_OPT_PASS_CLASS(ConvertSwitchReplacement)
REGISTER_OPT_PASS_CLASS(ExpandDumpFlag)
REGISTER_OPT_PASS_CLASS(ExpandMetaFg)
REGISTER_OPT_PASS_CLASS(UpdatestateDependEliminater)
REGISTER_OPT_PASS_CLASS(UpdatestateAssignEliminater)
REGISTER_OPT_PASS_CLASS(UpdatestateLoadsEliminater)
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
