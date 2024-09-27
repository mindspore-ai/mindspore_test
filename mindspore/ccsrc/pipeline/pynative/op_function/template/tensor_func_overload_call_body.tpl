py::object Tensor${class_name}(const py::args &args) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}});)
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param < std::string > (MS_CTX_DEVICE_TARGET);
  switch (sig.index_) {
    ${dispatch_cases}
  }
}

