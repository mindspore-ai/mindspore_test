/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <list>

#include "pybind11/pybind11.h"

#include "ir/anf.h"
#include "ir/tensor.h"
#include "pipeline/jit/ps/action.h"
#include "abstract/abstract_value.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/transform.h"
#include "pipeline/jit/ps/base.h"
#include "frontend/parallel/strategy.h"
#include "include/common/visible.h"
#include "include/fork_utils.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
// namespace to support pipeline structures definition
namespace pipeline {

namespace py = pybind11;

constexpr auto kActualArgumentIndex = "argument_index";

class Pipeline {
 public:
  Pipeline(const ResourcePtr &res, const std::vector<ActionItem> &actions) : resource_(res), actions_(actions) {}

  ~Pipeline() = default;

  void Run();

  ResourcePtr resource() { return resource_; }

  bool NeedCreateBackend();

 private:
  ResourcePtr resource_;
  std::vector<ActionItem> actions_;
};

class FRONTEND_EXPORT ExecutorPy : public std::enable_shared_from_this<ExecutorPy> {
 public:
  ExecutorPy() = default;
  virtual ~ExecutorPy() = default;
  bool Compile(const py::object &source, const py::tuple &args, const py::dict &kwargs, const py::object &phase);
  py::object Run(const py::tuple &args, const py::object &phase);
  void set_enable_tuple_broaden(bool enable_tuple_broaden) { enable_tuple_broaden_ = enable_tuple_broaden; }
  // Generate a key for mapping function graph
  py::object GenerateArgumentsKey(const py::object &obj, const py::tuple &args, const py::dict &kwargs,
                                  bool enable_tuple_broaden = false);
  void ClearCompileArgumentsResource();
  void SetJitConfig(const py::dict &jit_config);
  virtual void CleanCompileRes(const ResourcePtr &resource) = 0;
  FuncGraphPtr GetFuncGraph(const std::string &phase);
  void SetJitPrimalFuncGraph(const FuncGraphPtr &primal_func_graph, const std::string &phase);
  FuncGraphPtr GetJitPrimalFuncGraph(const std::string &phase);
  FuncGraphPtr GetJitGradGraph(const std::string &phase);
  void SetJitGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase);
  py::dict GetParams(const std::string &phase);
  bool HasCompiled(const std::string &phase) const;
  void DelNetRes(const py::object &source, const py::set &id);
  const std::string &phase() const { return phase_; }
  void set_queue_name(const std::string &queue_name) { queue_name_ = queue_name; }
  std::string get_queue_name(const std::string &dataset_phase);
  void set_compile_cache_dep_files(const py::list &compile_cache_dep_files) {
    compile_cache_dep_files_ = compile_cache_dep_files;
  }
  void set_weights_values(const py::dict &weights) { weights_ = weights; }
  // Check consistency of two arguments for mapping function graph
  void CheckArgumentsConsistency(const py::tuple &compile_args, const py::tuple &args_list, const py::object &target);
  py::bytes GetFuncGraphProto(const std::string &phase, const std::string &ir_type, const bool &incremental);
  virtual bool CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs,
                            const std::string &phase, bool trace_flag) = 0;
  bool executor_running() const { return executor_running_; }
  const std::string &obj_desc() const { return obj_desc_; }
  int32_t max_call_depth() const { return max_call_depth_; }
  void set_max_call_depth(int32_t max_call_depth) { max_call_depth_ = max_call_depth; }
  void ClearInfo();

 protected:
  virtual bool CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                            const py::object &phase) = 0;
  virtual py::object RunInner(const py::tuple &args, const py::object &phase) = 0;
  virtual void DelOneNetRes(const py::handle &py_phase) = 0;
  virtual void SaveCompiledGraph(const std::string &phase) = 0;
  ResourcePtr GetResource(const std::string &phase);
  void ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list);
  compile::VmEvalFuncPtr GetVmEvalFunc(const std::string &phase, const std::string &kind = kOutput);
  void ClearRunArgumentsResource(size_t input_arg_size, VectorRef *arg_list);
  // If enable compile cache, get the compile cache resource.
  void InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase);
  void InitCompileCacheResource(const ResourcePtr &resource, const std::string &phase);
  void set_process_id();

  std::map<std::string, ExecutorInfoPtr> info_;
  std::string phase_;
  std::string source_;
  std::string obj_desc_;
  bool enable_tuple_broaden_{false};
  std::map<PyObject *, std::pair<ValuePtr, AbstractBasePtr>> cur_convert_input_;

 private:
  void ClearCurConvertInput();
  void ReleaseResourceOnException(const py::object &phase);

  std::string queue_name_;
  py::list compile_cache_dep_files_;
  py::dict weights_;
  bool executor_running_{false};
  bool compile_cache_consistent_{true};
  int32_t max_call_depth_{-1};
  pid_t process_id_{0};
};
using ExecutorPyPtr = std::shared_ptr<ExecutorPy>;

// A function pipeline.
class FRONTEND_EXPORT GraphExecutorPy : public ExecutorPy {
 public:
  static std::shared_ptr<GraphExecutorPy> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<GraphExecutorPy>(new (std::nothrow) GraphExecutorPy());
    }
    executor_->set_process_id();
    return executor_;
  }

  ~GraphExecutorPy() override;

  bool CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs, const std::string &phase,
                    bool trace_flag) override;

  void ConvertArgs(const py::tuple &args, const py::dict &kwargs, bool is_auto_parallel,
                   abstract::AbstractBasePtrList *args_abs, std::vector<ValuePtr> *arguments);
  void ConvertSymbolicShape(const py::tuple &args, AbstractBasePtrList *args_abs);
  py::bytes GetOptimizeGraphProto(const std::string &phase);

  void BuildGraph(const py::dict &init_params, const std::string &phase) const;
  void ExportGraph(const std::string &file_name, const std::string &phase, const py::object encrypt = py::none(),
                   char *key = nullptr);
  py::bytes GetRandomStatus(const std::string &phase) const;
  void UpdataParamNodeDefaultInput(const std::string &phase,
                                   const std::unordered_map<std::string, py::object> &params_value);
  void PyExePath(const py::object &py_exe_path) const;
  void KernelBuildServerDir(const py::object &kernel_build_server_dir) const;
  py::dict GetParameterLayout(const std::string &phase);
  py::tuple FlopsCollection(const std::string &phase);
  // Get CNode name, input node name and attribute from each graph
  py::dict GetParallelGraphInfo(const std::string &phase);
  py::dict GetCNodeStrategy(const std::string &phase);
  py::list GetParallelParameterNameList(const std::string &phase);
  void SetCNodeStrategy(const std::string &name, const parallel::Strategies &strategy);
  size_t GetNumOpsInfo(const std::string &phase);
  void SetNumOpsInfo(size_t num_ops);
  py::dict GetAllreduceFusion(const std::string &phase);
  static void ClearRes();
  void SetOptimizeConfig(const py::list &optimize_cfg);
  std::string GetOptimizeConfig();
  void SetConfigPasses(const py::list &passes);
  py::list GetRunningPasses();

  void ParentBeforeFork();
  void ParentAfterFork();
  void ChildAfterFork();

  void CleanCompileRes(const ResourcePtr &resource) override;

 private:
  GraphExecutorPy() = default;
  void ParallelPostProcess(const string &phase, bool use_compile_cache);

  void ConvertObjectToTensors(const py::dict &dict,
                              std::map<std::string, std::shared_ptr<tensor::Tensor>> *const tensors,
                              const FuncGraphPtr &anf_graph) const;

  void DelOneNetRes(const py::handle &py_phase) override;
  bool CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                    const py::object &phase) override;
  py::object RunInner(const py::tuple &args, const py::object &phase) override;
  void SaveCompiledGraph(const std::string &phase) override;
#ifdef ENABLE_DEBUGGER
  void TerminateDebugger();
#endif

  static std::shared_ptr<GraphExecutorPy> executor_;
  static std::mutex instance_lock_;
  std::map<std::string, py::dict> stra_dict_;
  std::map<std::string, size_t> phase_to_num_op_info_;
};
using GraphExecutorPyPtr = std::shared_ptr<GraphExecutorPy>;

class JitCompilingScope {
 public:
  JitCompilingScope() { MsContext::GetInstance()->set_jit_status(kJitCompiling); }
  ~JitCompilingScope() { MsContext::GetInstance()->set_jit_status(kNotJit); }
};

class JitRunningScope {
 public:
  JitRunningScope() { MsContext::GetInstance()->set_jit_status(kJitRunning); }
  ~JitRunningScope() { MsContext::GetInstance()->set_jit_status(kNotJit); }
};

std::string GetJitLevel();

std::string GetObjDesc(const py::object &source);
bool IsPhaseLoadFromMindIR(const std::string &phase);
FRONTEND_EXPORT void CheckArgsValid(const py::object &source, const py::tuple &args);
FRONTEND_EXPORT py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs);

bool InitDistribute(const std::map<std::string, std::string> &options);

FRONTEND_EXPORT void ResetOpId();
FRONTEND_EXPORT void ResetOpIdWithOffset();
FRONTEND_EXPORT void InitHccl();
FRONTEND_EXPORT void FinalizeHccl();
FRONTEND_EXPORT uint32_t GetHcclRankId();
FRONTEND_EXPORT uint32_t GetHcclRankSize();
FRONTEND_EXPORT void InitPipeline();
void FinalizeBackend();

void CloseTsd(bool force = false);
FRONTEND_EXPORT void BindDeviceCtx();

FRONTEND_EXPORT FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                                        const std::string &dec_mode, const py::object decrypt = py::none());

FRONTEND_EXPORT FuncGraphPtr SplitMindIR(const std::string &file_name);

FRONTEND_EXPORT FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id,
                                                bool sapp);

// init and exec dataset sub graph
bool ME_EXPORT InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                               const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                               const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run);

// Build and run dataset subgraph for ms backend
bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run);

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list);

FRONTEND_EXPORT py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len,
                                    const std::string &enc_mode);
FRONTEND_EXPORT py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len,
                                    const std::string &dec_mode);
FRONTEND_EXPORT py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len,
                                        const std::string &dec_mode);
FRONTEND_EXPORT bool PyIsCipherFile(const std::string &file_path);
FRONTEND_EXPORT void FinalizeCluster();
FRONTEND_EXPORT void SwapCache(const py::object &host_, const py::object &device_, const py::object &block_mapping_,
                               const bool &type);

bool IsPhaseExport(const std::string &phase);
py::object BaseRefToPyDataWithUserData(const BaseRef &value, const AbstractBasePtr &abs);
void SetLoopCount(const ResourcePtr &resource);
void ResetId(const ResourcePtr &resource);
#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name);
void RecordIR(const size_t action_index, const size_t action_size, const std::string &action_name,
              const FuncGraphPtr &graph, FuncGraphPtr *user_graph);
#endif
AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden = false);
void AddManagerForFuncGraphArgs(const ResourcePtr &resource, const ValuePtrList &arguments);
void CheckInterpretNodeLineInfos();
void SetHookForArgAbstract(const py::object &arg, abstract::AbstractBasePtr abs);
FRONTEND_EXPORT bool RunJitPipeline();
FRONTEND_EXPORT void PreJit(const py::object &args, const py::object &kwargs);
FRONTEND_EXPORT void CleanCache();
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
