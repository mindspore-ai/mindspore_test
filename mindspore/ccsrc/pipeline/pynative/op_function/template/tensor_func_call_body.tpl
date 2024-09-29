py::object Tensor${class_name}(const py::args &py_args, const py::kwargs &py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}
  });
  py::list arg_list;
  py::list args(py_args);
  auto sig = parser.parse(py_args, py_kwargs, arg_list);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
  ${device_dispatcher}
}

