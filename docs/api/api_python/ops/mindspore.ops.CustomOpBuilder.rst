mindspore.ops.CustomOpBuilder
=============================

.. py:class:: mindspore.ops.CustomOpBuilder(name, sources, backend=None, include_paths=None, cflags=None, ldflags=None, **kwargs)

    `CustomOpBuilder` 用于初始化和配置MindSpore的自定义算子。用户可以通过该类定义和加载自定义算子模块，并将其应用到网络中。

    一般情况下，用户仅需在类的构造函数中传入源文件和额外的编译选项，并调用 `load` 接口即可完成算子的编译和加载。但如果用户有特殊的定制需求，也可以通过继承该类并重写部分接口来实现。需要注意的是，重写接口后，构造函数中传入的部分参数可能会被忽略。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **name** (str) - 自定义算子模块的唯一名称，用于标识算子。
        - **sources** (Union[str, list[str]]) - 自定义算子的源文件，可以是单个文件路径或文件路径列表。
        - **backend** (str, 可选) - 自定义算子的目标后端，例如 "CPU" 或 "Ascend"。默认值： ``None`` 。
        - **include_paths** (list[str], 可选) - 编译过程中需要的额外包含路径。默认值： ``None`` 。
        - **cflags** (str, 可选) - 编译过程中使用的额外C++编译选项。默认值： ``None`` 。
        - **ldflags** (str, 可选) - 链接过程中使用的额外链接选项。默认值： ``None`` 。
        - **kwargs** (dict, 可选) - 额外的关键字参数，用于扩展功能或自定义需求。

    .. note::
        - 如果提供了 `backend` 参数，编译和链接步骤中会自动添加支持目标后端的默认编译和链接选项。默认选项可参考 `CustomOpBuilder <https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/operations/custom_ops.py>`_ 代码中 `get_cflags` 和 `get_ldflags` 接口的实现。
        - `sources` 参数必须指向有效的自定义算子源文件。

    .. py:method:: build()

        编译自定义算子模块。

        该方法会根据提供的源文件、包含路径、编译选项和链接选项，生成自定义算子的动态库文件。

        返回：
            str，编译生成的模块文件路径。

    .. py:method:: get_cflags()

        获取编译自定义算子时的C++编译选项。

        返回：
            list[str]，C++编译选项列表。

    .. py:method:: get_include_paths()

        获取编译自定义算子时所需的头文件包含路径。

        返回：
            list[str]，包含路径的列表。

    .. py:method:: get_ldflags()

        获取链接自定义算子时的链接选项。

        返回：
            list[str]，链接选项列表。

    .. py:method:: get_sources()

        返回自定义算子的源文件路径。

        返回：
            Union[str, list[str]]，自定义算子的源文件路径，可能是字符串或字符串列表。

    .. py:method:: load()

        编译并加载自定义算子模块。

        返回：
            Module，加载的自定义算子模块。
