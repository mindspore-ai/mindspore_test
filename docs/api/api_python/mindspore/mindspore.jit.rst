mindspore.jit
=============

.. py:function:: mindspore.jit(function: Optional[Callable] = None, *, capture_mode: str = "ast", jit_level: str = "O0", dynamic: bool = False, fullgraph: bool = False, backend: Union[str, Callable] = "default", **options)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    参数：
        - **function** (Function, 可选) - 要编译成图的Python函数。默认值：``None``。
        - **capture_mode** (str, 可选) - 创建一张可调用的MindSpore图的方式，可选值有 ``"ast"`` 、 ``"bytecode"`` 和 ``"trace"`` 。默认值： ``"ast"``。

          - `ast <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph.html>`_ ：解析Python的ast以构建静态图。
          - `bytecode <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/pynative.html#pijit>`_ ：在运行时解析Python字节码以构建静态图。这是一个实验特性，可能会被更改或者删除。
          - `trace` ：追踪Python代码的执行以构建静态图。这是一个实验特性，可能会被更改或者删除。

        - **jit_level** (str, 可选) - 控制编译优化的级别。可选值有 ``"O0"`` 、 ``"O1"`` 和 ``"O2"`` 。默认值： ``"O0"``。
        - **dynamic** (bool, 可选) - 是否需要进行动态shape编译。当前为预留参数，暂不起作用。默认值： ``False``。
        - **fullgraph** (bool, 可选) - 是否捕获整个函数来编译成图。如果为False，jit会尽可能地尝试兼容函数中的所有Python语法。如果为True，则需要整个函数都可以被捕获成图，否则（即有不支持的Python语法），会抛出一个异常。当前只对capture_mode为 ``"ast"`` 时生效。默认值： ``False``。
        - **backend** (str, 可选) - 使用的编译后端。默认值： ``"default"`` 。
        - **\*\*options** (dict) - 传给编译后端的选项字典。

    返回：
        函数，如果 `fn` 不是None，则返回一个已经将输入 `fn` 编译成图的可执行函数；如果 `fn` 为None，则返回一个装饰器。当这个装饰器使用单个 `fn` 参数进行调用时，等价于 `fn` 不是None的场景。
