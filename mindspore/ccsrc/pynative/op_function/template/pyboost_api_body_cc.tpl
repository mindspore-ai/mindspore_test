py::object PYNATIVE_EXPORT ${func_name}_Base(const PrimitivePtr &prim, const py::list &args) {
#ifndef ENABLE_TEST
  static Converter converter(&ops::${op_def_name});
  converter.Parse(args);
  ${parser_body}
  return ${func_name}_OP(prim, converter.source_type(), ${op_args});
#else
  return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
#endif
}

