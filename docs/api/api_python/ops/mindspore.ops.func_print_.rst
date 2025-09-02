mindspore.ops.print\_
=====================

.. py:function:: mindspore.ops.print_(*input_x)

    将输入数据进行打印输出。

    默认打印在屏幕上。也可以通过 `context` 设置 `print_file_path` 参数保存在指定文件中。获取更多信息，请查看 :func:`mindspore.set_context`。

    在Ascend平台上的Graph模式下，可以通过设置环境变量 `MS_DUMP_SLICE_SIZE` 和 `MS_DUMP_WAIT_TIME` 解决在输出大tensor或输出tensor比较密集场景下算子执行失败的问题。

    .. note::
        在PyNative模式下，请使用Python print函数。在Ascend平台上的Graph模式下，bool、int、float、tuple以及list类型数据将被转换为tensor进行打印，str保持不变。

    参数：
        - **input_x** (Union[Tensor, bool, int, float, str, tuple, list]) - print_的输入。支持多个输入，用'，'分隔。

    返回：
        无效返回值，应忽略。
