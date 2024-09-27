py::object ${func_name}_Base(const PrimitivePtr &prim, const py::list &args) {
  #ifndef ENABLE_TEST
    MS_LOG(DEBUG) << "Run ${func_name} start";
    auto op_run_info = PyNativeAlgo::PyBoost::Init(prim, args);
    op_run_info->signatures = ops::${op_def_name}.signatures_;
    static Converter converter(&ops::${op_def_name});
    converter.Parse(args);
    ${parser_body}

    static auto top_type = PredictOutType(op_run_info);
    auto node = stub::MakeTopNode(top_type);
    auto comm_handle_py = std::make_shared<hal::CommHandlePy>(runtime::OpRunner::GetDeviceContext(op_run_info->base_op_run_info.device_target));
    auto comm_handle_py_obj = py::cast(comm_handle_py);
    const auto &output_obj = py::make_tuple(node.first, comm_handle_py_obj);    
    kernel::pyboost::CommHandlePtr comm_handle{nullptr};
    bool is_ascend = op_run_info->base_op_run_info.device_target == kAscendDevice;
    if (!is_ascend) {
      MS_LOG(EXCEPTION) << "mindspore.communication.comm_func is not supported in CPU or GPU. Please use "
                        << "Operators in mindspore.communication.comm_ops.py";
    }

    comm_handle_py->comm_handle()->CreateEvent();
    comm_handle = comm_handle_py->comm_handle();
    
    GilReleaseWithCheck release_gil;
    op_run_info->stub_output = node.second;
    op_run_info->source_type = converter.source_type();
    DispatchOp(
      std::make_shared<FrontendTask>(
        [${op_args}, comm_handle](const FrontendOpRunInfoPtr &op_run_info) {
          MS_LOG(DEBUG) << "Run frontend task ${func_name} start";
          auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
          kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->base_op_run_info.stream_id);

          // stub tensor to tensor.
          ${convert_stub}

          // Create op
          auto op = CREATE_PYBOOST_OP(${op_name}, op_run_info->base_op_run_info.device_target);
          op->set_comm_handle(comm_handle);
          const auto &op_prim = op->primitive();

          // Do mixed precision and implicit cast
          static const std::vector<std::vector<size_t>> same_type_table{${same_type}};
          auto [${cast_args}] = PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<${type_num}>(op_run_info, same_type_table, ${call_args});

          // Run op
          (void)op->Call(${cast_args});
          ${optional_to_value}

          // Data sync in mix mode(Graph and PyNative)
          PyNativeAlgo::PyBoost::DataSyncForGraph(op, {${grad_args}});

          // Update op and op_run_info by op outputs
          PyNativeAlgo::PyBoost::UpdateOpRunInfo(op, op_run_info);

          // Do auto grad
          if (op_run_info->requires_grad) {
            // Refresh op prim, otherwish the size of inputs will be incorrect.
            op_run_info->op_grad_info->op_prim = op_prim;
            PyNativeAlgo::PyBoost::DoGrad(op, op_run_info, {${grad_args}});
          }
          kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);

          MS_LOG(DEBUG) << "Run frontend task ${func_name} end";
        },
        op_run_info
      )
    );
    MS_LOG(DEBUG) << "Run ${func_name} end";
    return output_obj;
  #else
    return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
  #endif
}

py::object ${func_name}(const py::args &args) {
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
                      << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     prim->name(), false, true);
  return ${func_name}_Base(prim, args[1]);
}

class ${class_name}PrimAdapter: public PrimitiveFunctionAdapter {
  public:
   ${class_name}PrimAdapter() : PrimitiveFunctionAdapter() {}
   ~${class_name}PrimAdapter() = default;
   std::string name() override { return "${class_name}"; }
   py::object Call(const py::args &args) {
     runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                        "${class_name}", false, true);
     return ${func_name}_Base(prim::kPrim${class_name}, args);
   }
};
