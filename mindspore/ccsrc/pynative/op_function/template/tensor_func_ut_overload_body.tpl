py::function fn;
py::object res;
switch (sig.index_) {
  ${ut_dispatch_cases}
}
return res;