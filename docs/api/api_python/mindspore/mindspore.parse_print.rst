mindspore.parse_print
=======================================

.. py:function:: mindspore.parse_print(print_file_name)

    解析由 :class:`mindspore.ops.Print` 生成的数据文件。

    .. note::
        这个接口在2.5版本废弃，并且会在未来版本移除。

    参数：
        - **print_file_name** (str) - 需要解析的文件名。

    返回：
        List，由Tensor组成的list。

    异常：
        - **ValueError** - 指定的文件不存在或为空。
        - **RuntimeError** - 解析文件失败。
