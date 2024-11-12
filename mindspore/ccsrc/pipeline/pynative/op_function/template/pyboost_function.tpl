py::object ME_EXPORT ${func_name}_Base(const PrimitivePtr &prim, const py::list &args) {
  #ifndef ENABLE_TEST
    MS_LOG(DEBUG) << "Run ${func_name} start";
    auto op_run_info = PyNativeAlgo::PyBoost::Init(prim, args);
    op_run_info->signatures = ops::${op_def_name}.signatures_;
    static Converter converter(&ops::${op_def_name});
    converter.Parse(args);
    ${parser_body}

    static auto top_type = PredictOutType(op_run_info);
    auto node = stub::MakeTopNode(top_type);
    {
      GilReleaseWithCheck release_gil;
      op_run_info->stub_output = node.second;
      op_run_info->source_type = converter.source_type();
      DispatchOp(
        std::make_shared<FrontendTask>(
          [${op_args}, prim](const FrontendOpRunInfoPtr &op_run_info) {
            MS_LOG(DEBUG) << "Run frontend task ${func_name} start";
            auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
            kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->base_op_run_info.stream_id);

            // stub tensor to tensor.
            ${convert_stub}

            // Do mixed precision and implicit cast
            static const std::vector<std::vector<size_t>> same_type_table{${same_type}};
            auto [${cast_args}] = PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<${type_num}>(op_run_info, same_type_table, ${call_args});

            kernel::pyboost::OpRunStatus::Get().set_run_info(
                kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                          op_run_info->async_status.is_jit_compiling,
                                          op_run_info->async_status.custom_bprop_cell_count,
                                          op_run_info->base_op_run_info.device_target));
            kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);

            auto outputs = kernel::pyboost::${operator_name}(${cast_args});
            auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();

            // Tensor/tuple<Tensor> to ValuePtr
            auto real_out = PyNativeAlgo::PyBoost::${is_multi}OutputToValue(outputs);
            // Set output value to python
            PyNativeAlgo::PyBoost::UpdateStubOutput(op, op_run_info->stub_output, op->output_abs(), real_out);
            // Data sync in mix mode(Graph and PyNative)
            PyNativeAlgo::PyBoost::DataSyncForGraph(op);
            kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);

            MS_LOG(DEBUG) << "Run frontend task ${func_name} end";
          },
          op_run_info
        )
      );
      MS_LOG(DEBUG) << "Run ${func_name} end";
    }
    return node.first;
  #else
    return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
  #endif
}

py::object ME_EXPORT ${func_name}(const py::args &args) {
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
                      << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     prim->name(), false, true);
  auto res = ${func_name}_Base(prim, args[1]);
  trace::Capture(args, res);
  return res;
}

class ME_EXPORT ${class_name}PrimAdapter: public PrimitiveFunctionAdapter {
  public:
   ${class_name}PrimAdapter() : PrimitiveFunctionAdapter() {}
   ~${class_name}PrimAdapter() = default;
   std::string name() override { return "${class_name}"; }
   py::object Call(const py::list &args) {
     runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                        "${class_name}", false, true);
     auto res = ${func_name}_Base(prim::kPrim${class_name}, args);
     trace::Capture(args, res, "${class_name}");
     return res;
   }
};
