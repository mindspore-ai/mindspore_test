PYNATIVE_EXPORT PyObject* ${func_name}_Base(const PrimitivePtr &prim, PyObject* args) {
#ifndef ENABLE_TEST
  ${mark_side_effect}
  static pynative::Converter converter(&ops::${op_def_name});
  converter.Parse(args);
  ${parser_body}
  return ${func_name}_OP(prim, converter.source_type(), ${op_args});
#else
  py::object py_args = py::reinterpret_borrow<py::object>(args);
  py::object res = PyNativeAlgo::PyBoost::RunPyFunction(prim, py_args);
  return res.release().ptr();
#endif
}

