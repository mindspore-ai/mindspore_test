mindspore.ops.Custom
=====================

.. py:class:: mindspore.ops.Custom(func, out_shape=None, out_dtype=None, func_type="pyfunc", bprop=None, reg_info=None)

    `Custom` 算子是MindSpore自定义算子的统一接口。用户可以利用该接口自行定义MindSpore内置算子库尚未包含的算子。
    根据输入函数的不同，用户可以创建多个自定义算子，并将其应用于神经网络中。
    关于自定义算子的详细说明和介绍，包括参数的正确书写，请参见 `自定义算子教程 <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/op_custom.html>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        不同自定义算子的函数类型（func_type）支持的平台类型不同。每种类型支持的平台如下：

        - "aot": ["GPU", "CPU", "Ascend"].
        - "pyfunc": ["CPU"].

    参数：
        - **func** (Union[function, str]) - 自定义算子的函数表达。

          - function：如果 `func` 是函数类型，那么 `func` 应该是一个Python函数，它描述了用户定义的操作符的计算逻辑。

          - 字符串：如果 `func` 是字符串类型，那么 `str` 应该是包含函数名的文件路径。当 `func_type` 是"aot"时，可以使用这种方式。

            对于"aot"：

            a) GPU/CPU（仅Linux）平台

            "aot"意味着提前编译。在这种情况下，Custom直接启动用户定义的"xxx.so"文件作为操作符。用户需要提前将手写的"xxx.cu"/"xxx.cc"文件编译成"xxx.so"，并提供文件路径和函数名。

            - "xxx.so"文件生成：

              1) GPU平台：给定用户定义的"xxx.cu"文件（例如"{path}/add.cu"），使用nvcc命令进行编译（例如 :code:`nvcc --shared -Xcompiler -fPIC -o add.so add.cu`）。

              2) CPU平台：给定用户定义的"xxx.cc"文件（例如"{path}/add.cc"），使用g++/gcc命令进行编译（例如 :code:`g++ --shared -fPIC -o add.so add.cc`）。

            - 定义"xxx.cc"/"xxx.cu"文件：

              "aot"是一个跨平台的标识符。"xxx.cc"或"xxx.cu"中定义的函数具有相同的参数。通常，该函数应该像这样：

              .. code-block::

                  int func(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra)

              参数：

              - `nparam(int)` : 输入和输出的总数；假设操作符有2个输入和3个输出，那么 `nparam=5` 。
              - `params(void **)` : 输入和输出指针的数组指针；输入和输出的指针类型为 `void *` ；假设操作符有2个输入和3个输出，那么第一个输入的指针是 `params[0]` ，第二个输出的指针是 `params[3]` 。
              - `ndims(int *)` : 输入和输出维度数的数组指针；假设 `params[i]` 是一个1024x1024的张量， `params[j]` 是一个77x83x4的张量，那么 `ndims[i]=2` ， `ndims[j]=3` 。
              - `shapes(int64_t **)` : 输入和输出形状（ `int64_t *` ）的数组指针；第 `i` 个输入的第 `j` 个维度的大小是 `shapes[i][j]` （其中 `0<=j<ndims[i]` ）；假设 `params[i]` 是一个2x3的张量， `params[j]` 是一个3x3x4的张量，那么 `shapes[i][0]=2` ， `shapes[j][2]=4` 。
              - `dtypes(const char **)` : 输入和输出类型（ `const char *` ）的数组指针；（例如："float32"、"float16"、"float"、"float64"、"int"、"int8"、"int16"、"int32"、"int64"、"uint"、"uint8"、"uint16"、"uint32"、"uint64"、"bool"）
              - `stream(void *)` : 流指针，仅在CUDA文件中使用。
              - `extra(void *)` : 用于进一步扩展。

              返回值（int）：

              - 0: 如果这个AOT内核成功执行，MindSpore将继续运行。
              - 其他值: MindSpore将引发异常并退出。

              示例：详见 `tests/st/ops/graph_kernel/custom/aot_test_files/` 中的详细信息。

            - 在Custom中使用：

              .. code-block::

                  Custom(func="{dir_path}/{file_name}:{func_name}", ...)

              例如：Custom(func="./reorganize.so:CustomReorganize", out_shape=[1], out_dtype=mstype.float32, "aot")

            b) Ascend平台

            在Ascend平台使用Custom算子之前，用户首先需要基于Ascend C开发自定义算子并编译。完整的开发和使用流程可参考教程 `AOT类型自定义算子（Ascend平台） <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_custom_ascendc.html>`_。
            在入参 `func` 中传入算子的名字。根据infer函数的实现方式，存在以下两种使用方式：

            - **Python infer**：若算子的infer函数是Python实现，即通过 `out_shape` 和 `out_dtype` 参数传入infer shape和infer type函数，则指定 `func="CustomName"` 。
            - **C++ infer**：若算子的infer函数通过C++实现，则在func中传入infer shape或infer type实现文件的路径并用 `:` 隔开算子名字，例如： `func="add_custom_infer.cc:AddCustom"` 。

        - **out_shape** (Union[function, list, tuple], 可选) - 自定义算子的输出的形状或者输出形状的推导函数。默认值： ``None`` 。
        - **out_dtype** (Union[function, :class:`mindspore.dtype`, tuple[:class:`mindspore.dtype`]], 可选) - 自定义算子的输出的数据类型或者输出数据类型的推导函数。默认值： ``None`` 。
        - **func_type** (str, 可选) - 自定义算子的函数类型，必须是[ ``"aot"`` , ``"pyfunc"``]中之一。默认值： ``"pyfunc"``。
        - **bprop** (function, 可选) - 自定义算子的反向函数。默认值： ``None``。
        - **reg_info** (Union[str, dict, list, tuple], 可选) - 自定义算子的算子注册信息。默认值： ``None`` 。

    输入：
        - **input** (Union(tuple, list)) - 输入要计算的Tensor。

    输出：
        Tensor，自定义算子的计算结果。

    异常：
        - **TypeError** - 如果输入 `func` 不合法，或者 `func` 对应的注册信息类型不对。
        - **ValueError** - `func_type` 的值不在列表内。
        - **ValueError** - 算子注册信息不合法，包括支持平台不匹配，算子输入和属性与函数不匹配。