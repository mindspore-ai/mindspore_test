class ${class_name}Functional : public Functional {
 public:
  ${class_name}Functional() : Functional("${func_name}", false) {};
  ~${class_name}Functional() = default;
  py::object Call(const py::object &self, const py::args &args, const py::kwargs &kwargs) {
    static PythonArgParser parser({
    ${signatures}
      }, "${func_name}");
    py::list arg_list;
    auto sig = parser.parse(args, kwargs, &arg_list, true);
    arg_list.insert(${self_index}, self);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
    #ifndef ENABLE_TEST
      switch (sig.index_) {
        ${dispatch_cases}
      }
      return py::none();
    #else
      ${ut_overload_body}
    #endif
  }
};

