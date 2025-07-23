PyObject* TensorMethod${cpp_func_name}(PyObject* self, PyObject* py_args, PyObject* py_kwargs) {
  static mindspore::pynative::PythonArgParser parser({
    ${signatures}
  }, "${func_name}");
  auto input_tensor = mindspore::pynative::UnpackTensor(self, "${func_name}");
  auto parse_args = parser.Parse(py_args, py_kwargs, true);
  parse_args.InsertInputTensor(${self_index}, self);

  auto backend = DeviceManagerConf::GetInstance()->device_type();
  #ifndef ENABLE_TEST
    ${device_dispatcher}
    Py_RETURN_NONE;
  #else
    ${ut_body}
  #endif
}

