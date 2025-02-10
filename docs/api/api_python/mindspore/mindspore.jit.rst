mindspore.jit
=============

.. py:function:: mindspore.jit(function: Optional[Callable] = None, *, capture_mode: str = "ast", jit_level: str = "O0", dynamic: bool = False, fullgraph: bool = False, backend: str = "", **options)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    .. note::
        - 不支持在静态图模式下，运行带装饰@jit(capture_mode="bytecode")的函数，此时该装饰@jit(capture_mode="bytecode")视为无效。
        - 不支持在@jit(capture_mode="ast")装饰的函数内部调用带装饰@jit(capture_mode="bytecode")的函数，该装饰@jit(capture_mode="bytecode")视为无效。

    参数：
        - **function** (Function, 可选) - 要编译成图的Python函数。默认值：``None``。

    关键字参数：
        - **capture_mode** (str, 可选) - 创建一张可调用的MindSpore图的方式，可选值有 ``"ast"`` 、 ``"bytecode"`` 和 ``"trace"`` 。默认值： ``"ast"``。

          - `ast <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph.html>`_ ：解析Python的ast以构建静态图。
          - `bytecode <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/pynative.html#pijit>`_ ：在运行时解析Python字节码以构建静态图。这是一个实验特性，可能会被更改或者删除。
          - `trace` ：追踪Python代码的执行以构建静态图。这是一个实验特性，可能会被更改或者删除。

        - **jit_level** (str, 可选) - 控制编译优化的级别。目前仅在使用ms_backend后端时生效。可选值有 ``"O0"`` 和 ``"O1"`` 。默认值： ``"O0"``。

          - O0: 除必要影响功能的优化外，其他优化均关闭。
          - O1: 使能常用优化和自动算子融合优化。

        - **dynamic** (bool, 可选) - 是否需要进行动态shape编译。当前为预留参数，暂不起作用。默认值： ``False``。
        - **fullgraph** (bool, 可选) - 是否捕获整个函数来编译成图。如果为False，jit会尽可能地尝试兼容函数中的所有Python语法。如果为True，则需要整个函数都可以被捕获成图，否则（即有不支持的Python语法），会抛出一个异常。当前只对capture_mode为 ``"ast"`` 时生效。默认值： ``False``。
        - **backend** (str, 可选) - 使用的编译后端。如果该参数未被设置，框架默认Atlas训练系列产品为 ``GE`` 后端 ，默认其他产品包括Atlas A2训练系列产品为 ``ms_backend`` 后端。

          - ms_backend: 使用逐算子执行的执行方式。
          - GE: 使用下沉的执行方式，整个模型都会下沉到device侧执行，仅可作用于模型的最外层Cell。仅支撑昇腾硬件。

        - **\*\*options** (dict) - 传给编译后端的选项字典。

          某些配置适用于特定的设备和后端，有关详细信息，请参见下表：

          +---------------------------+---------------------------+-------------------------+
          |  配置项                   |  硬件平台支持             |  编译后端支持           |
          +===========================+===========================+=========================+
          | disable_format_transform  |  GPU                      |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+
          | exec_order                |  Ascend                   |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+
          | ge_options                |  Ascend                   |  GE                     |
          +---------------------------+---------------------------+-------------------------+
          | infer_boost               |  Ascend                   |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+

          - **disable_format_transform** (bool, 可选) - 表示是否取消NCHW到NHWC的自动格式转换功能。当fp16的网络性能不如fp32的时，可以设置 `disable_format_transform` 为 ``True`` ，以尝试提高训练性能。默认值： ``False`` 。
          - **exec_order** (str, 可选) - 算子执行时的排序方法，GRAPH_MODE(0)下jit_level为O0或者O1时生效。不同的执行顺序会使得网络的执行内存和性能有所差异，当前仅支持两种排序方法：bfs和dfs，默认方法为bfs。

            - bfs：默认的排序方法，广度优先排序，具备较好的通信掩盖效果，执行性能相对较好。
            - dfs：可选择的排序方法，深度优先排序，性能相对bfs执行序较差，但内存占用较少，建议在其他执行序OOM的场景下尝试dfs。

          - **ge_options** (dict) - 设置GE的options配置项，配置项分为 ``global`` 和 ``session`` 二类 。这是一个实验特性，可能会被更改或者删除。
            详细的配置请查询 `options配置说明 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendgraphapi/atlasgeapi_07_0146.html>`_ 。

            - global (dict): 设置global类的选项。
            - session (dict): 设置session类的选项。

          - **infer_boost** (str, 可选) - 用来使能推理模式。默认值为 ``“off”``，表示关闭。其值范围如下：

            - on: 开启推理模式，推理性能得到较大提升。
            - off: 关闭推理模式，使用前向运算进行推理，性能较差。

    返回：
        函数，如果 `fn` 不是None，则返回一个已经将输入 `fn` 编译成图的可执行函数；如果 `fn` 为None，则返回一个装饰器。当这个装饰器使用单个 `fn` 参数进行调用时，等价于 `fn` 不是None的场景。
