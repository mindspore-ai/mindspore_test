py::object TensorMethod${cpp_func_name}(const py::object &self, const py::args &py_args, const py::kwargs &py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}
  }, "${func_name}");
  py::list arg_list;
  auto sig = parser.Parse(py_args, py_kwargs, &arg_list, true);
  arg_list.insert(${self_index}, self);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
  #ifndef ENABLE_TEST
    ${device_dispatcher}
    return py::none();
  #else
    ${ut_body}
  #endif
}

