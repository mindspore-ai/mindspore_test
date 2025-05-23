mindspore.enable_dynamic
========================

.. py:function:: mindspore.enable_dynamic(**kwargs)

    用于指定参数的shape是动态shape或者动态rank。

    .. note::
        - 需要配合jit接口使用，不使用jit装饰器则无法使能动态shape和动态rank功能。
        - 在set_context(mode=GRAPH_MODE)和nn.Cell同时使用的场景下，使用enable_dynamic将会报错。

    参数：
        - **kwargs** (dict) - 输入类型为Tensor、tuple[Tensor]或list[Tensor]。如果需要指定参数的shape中一个或多个维度为动态shape，可以将shape中相应维度设置为None。如果需要生成指定参数的shape是动态rank，可以将shape设置为None。

    返回：
        函数，返回指定了参数动态shape信息的一个函数。
