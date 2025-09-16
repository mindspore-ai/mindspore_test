struct LayoutCacheKey {
    std::vector<std::string> layout_ids;

    bool operator==(const LayoutCacheKey& other) const {
        return layout_ids == other.layout_ids;
    }
};

struct LayoutCacheKeyHash {
    size_t operator()(const LayoutCacheKey& key) const {
        size_t seed = 0;
        for (const auto& id : key.layout_ids) {
            seed ^= std::hash<std::string>()(id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class LayoutCacheManager {
public:
    using LayoutCache = std::unordered_map<std::string,
        std::unordered_map<LayoutCacheKey, py::object, LayoutCacheKeyHash>>;

    using DistributedOpCache = std::unordered_map<std::string, py::object>;

    static LayoutCacheManager& GetInstance() {
        static LayoutCacheManager instance;
        return instance;
    }

    LayoutCache& GetLayoutCache() {
        return layout_cache_;
    }

    py::object GetDistributedOp(const std::string& op_name) {

        auto it = op_cache_.find(op_name);
        if (it != op_cache_.end()) {
            return it->second;
        }

        auto get_distributed_op = py::module_::import("mindspore.parallel.spmd.ops.parallel_ops_register")
                                    .attr("get_distributed_op");
        py::object op = get_distributed_op(op_name);
        op_cache_[op_name] = op;

        return op;
    }

    void ClearCache() {
        layout_cache_.clear();
        op_cache_.clear();
    }

private:
    LayoutCacheManager() {
        auto atexit = py::module_::import("atexit");
        atexit.attr("register")(py::cpp_function([this]() {
            this->ClearCache();
        }));
    }

    ~LayoutCacheManager() = default;

    LayoutCache layout_cache_;
    DistributedOpCache op_cache_;
};

template <typename Func, typename... Args>
PyObject* WithLayoutInfer(const PrimitivePtr &prim, Func &&func, PyObject* py_args, Args &&... args) {
    try {
        if (!py::isinstance<py::list>(py_args)) {
            MS_LOG(EXCEPTION) << "Input args is not a list.";
        }
        py::list py_args_list = py::cast<py::list>(py_args);
        auto& cache_manager = LayoutCacheManager::GetInstance();
        auto& layout_cache = cache_manager.GetLayoutCache()[prim->name()];
        py::object distribute_op = cache_manager.GetDistributedOp(prim->name());

        LayoutCacheKey cache_key;
        py::list input_layouts;
        py::list extra_args;
        bool contain_parallel_args = false;

        // Parse tuple tensor args
        py::list expanded_args;
        for (auto arg : py_args_list) {
            if (py::isinstance<py::tuple>(arg)) {
                py::tuple tuple_arg = py::cast<py::tuple>(arg);
                for (size_t j = 0; j < tuple_arg.size(); ++j) {
                    expanded_args.append(tuple_arg[j]);
                }
            } else {
                expanded_args.append(arg);
            }
        }

        // Collect layout and no layout args
        for (size_t i = 0; i < expanded_args.size(); ++i) {
            if (expanded_args[i].is_none()) {
                input_layouts.append(py::none());
                continue;
            }
            if (!py::hasattr(expanded_args[i], "_layout")) {
                py::object arg_str = py::str(expanded_args[i]);
                std::string id_str = py::cast<std::string>(arg_str);
                cache_key.layout_ids.push_back(id_str);
                extra_args.append(expanded_args[i]);
                input_layouts.append(py::none());
                continue;
            }
            contain_parallel_args = true;
            py::object layout = expanded_args[i].attr("_layout");
            py::object layout_id = layout.attr("compact_str");
            std::string id_str = py::cast<std::string>(py::str(layout_id));
            cache_key.layout_ids.push_back(id_str);
            input_layouts.append(layout);
        }

        if (!contain_parallel_args) {
            return std::forward<Func>(func)(std::forward<Args>(args)...);
        }

        py::object output_layout;
        auto it = layout_cache.find(cache_key);

        if (it != layout_cache.end()) {
            output_layout = it->second;
        } else {
            py::tuple all_args = py::make_tuple(input_layouts, extra_args);
            output_layout = distribute_op.attr("infer_layout")(*all_args);
            layout_cache[cache_key] = output_layout;
        }

        auto py_output = std::forward<Func>(func)(std::forward<Args>(args)...);
        if (py::isinstance<py::tuple>(py_output)) {
            py::tuple output_tuple = py::cast<py::tuple>(py_output);
            if (py::isinstance<py::tuple>(output_layout)) {
                py::tuple layout_tuple = py::cast<py::tuple>(output_layout);
                if (output_tuple.size() == layout_tuple.size()) {
                    for (size_t i = 0; i < output_tuple.size(); ++i) {
                        output_tuple[i].attr("_layout") = layout_tuple[i];
                    }
                } else {
                    MS_LOG(ERROR) << "Output tuple size (" << output_tuple.size()
                                  << ") does not match layout tuple size (" << layout_tuple.size() << ")";
                    throw std::runtime_error("Output and layout tuple size mismatch");
                }
            } else {
                MS_LOG(ERROR) << "Output is a tuple but layout is not";
                throw std::runtime_error("Output is tuple but layout is not");
            }
        } else {
            auto obj = py::reinterpret_borrow<py::object>(py_output);
            obj.attr("_layout") = output_layout;
        }

        return py_output;
    } catch (const py::error_already_set &e) {
        MS_LOG(ERROR) << "Python exception in layout inference: " << e.what();
        throw;
    } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Exception in layout inference: " << e.what();
        throw;
    }
}