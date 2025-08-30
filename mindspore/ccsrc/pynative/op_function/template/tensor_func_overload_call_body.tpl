PyObject* TensorMethod${cpp_func_name}(PyObject* self, PyObject* py_args, PyObject* py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}
  }, "${func_name}");
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     "tensor.${func_name}", false, true);
  auto input_tensor = mindspore::pynative::UnpackTensor(self, "${func_name}");
  auto parse_args = parser.Parse(py_args, py_kwargs, true);
  parse_args.InsertInputTensor(${self_index}, self);
  ${mark_side_effect}
  auto backend = DeviceManagerConf::GetInstance()->device_type();
  #ifndef ENABLE_TEST
    switch (parse_args.GetOvertLoadIndex()) {
      ${dispatch_cases}
    }
    Py_RETURN_NONE;
  #else
    ${ut_overload_body}
  #endif
}

