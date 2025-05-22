py::object PYNATIVE_EXPORT ${func_name}(const py::args &args) {
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
                      << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     prim->name(), false, true);
  auto res = ${func_name}_Base(prim, args[1]);
  trace::Capture(args, &res);
  return res;
}

class PYNATIVE_EXPORT ${class_name}PrimAdapter: public PrimitiveFunctionAdapter {
  public:
   ${class_name}PrimAdapter() : PrimitiveFunctionAdapter() {}
   ~${class_name}PrimAdapter() = default;
   std::string name() override { return "${class_name}"; }
   py::object Call(const py::list &args) {
     runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                        "${class_name}", false, true);
     auto res = ${func_name}_Base(prim::kPrim${class_name}, args);
     trace::Capture(args, prim::kPrim${class_name}, &res);
     return res;
   }
};
