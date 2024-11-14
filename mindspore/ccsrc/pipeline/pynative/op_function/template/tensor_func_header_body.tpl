class Tensor${class_name}Register {
 public:
  using PyBoostOp = std::function<py::object(const py::list &args)>;
  static void Register(const PyBoostOp &op) { op_ = op; }

  static const PyBoostOp &GetOp() { return op_; }

 private:
  inline static PyBoostOp op_;
};

