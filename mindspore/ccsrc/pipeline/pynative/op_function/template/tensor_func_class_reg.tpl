class ${class_name}Register {
 public:
  ${class_name}Register() {
    tensor::Tensor${class_name}Register::Register([](const py::list &args) {
      static ${class_name}PrimAdapter ${op_name}_prim;
      return ${op_name}_prim.Call(args);
    });
  }
};

static ${class_name}Register ${op_name}_reg;

