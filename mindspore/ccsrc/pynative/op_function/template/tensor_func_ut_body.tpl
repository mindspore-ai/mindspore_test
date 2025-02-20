MS_LOG(INFO) << "Callback python method in UT: ${py_method}";
py::function fn = python_adapter::GetPyFn("mindspore.ops.tensor_method", "${py_method}");
py::object res = fn(self, *py_args, **py_kwargs);
return res;