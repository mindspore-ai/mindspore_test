py::object TensorMethod${cpp_func_name}(const py::object &self, const py::args &py_args, const py::kwargs &py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}
  }, "${func_name}");
  auto input_tensor = mindspore::pynative::UnpackTensor(self, "${func_name}");
  auto parse_args = parser.Parse(py_args, py_kwargs, true);
  parse_args.src_types_.insert(parse_args.src_types_.begin(), ops::OP_DTYPE::DT_BEGIN);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
  #ifndef ENABLE_TEST
    switch (parse_args.GetOvertLoadIndex()) {
      ${dispatch_cases}
    }
    return py::none();
  #else
    ${ut_overload_body}
  #endif
}

