py::object Tensor${class_name}(const py::args &args) {
  MS_LOG(INFO) << "Call Tensor${class_name}";
  return ToPython(Tensor${class_name}Register::GetOp()(args));
}

