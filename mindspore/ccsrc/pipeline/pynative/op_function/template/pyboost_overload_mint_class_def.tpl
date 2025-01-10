class ${cpp_func_name}Functional : public Functional {
 public:
  ${cpp_func_name}Functional() : Functional("${func_name}") {};
  ~${cpp_func_name}Functional() = default;
  py::object Call(const py::args &args, const py::kwargs &kwargs) {
    static PythonArgParser parser({
    ${signatures}
      }, "${func_name}");
    py::list arg_list;
    auto sig = parser.Parse(args, kwargs, &arg_list, false);
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

static std::shared_ptr<${cpp_func_name}Functional> ${func_name}_instance = std::make_shared<${cpp_func_name}Functional>();

