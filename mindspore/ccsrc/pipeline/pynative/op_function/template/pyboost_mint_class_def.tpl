class ${class_name}Functional : public Functional {
 public:
  ${class_name}Functional() : Functional("${func_name}") {};
  ~${class_name}Functional() = default;
  py::object Call(const py::args &args, const py::kwargs &kwargs) {
    static PythonArgParser parser({
      ${signatures}
    }, "${func_name}");
    py::list arg_list;
    auto sig = parser.Parse(args, kwargs, &arg_list, false);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
    ${device_dispatcher}
    return py::none();
  }
};

static std::shared_ptr<${class_name}Functional> ${func_name}_instance = std::make_shared<${class_name}Functional>();

