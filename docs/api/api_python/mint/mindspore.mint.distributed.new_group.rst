mindspore.mint.distributed.new_group
=====================================

.. py:function:: mindspore.mint.distributed.new_group(ranks=None,timeout=None,backend=None,pg_options=None,use_local_synchronization=False,group_desc=None)

    创建用户自定义的通信组实例。

    .. note::
        `new_group` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。

    参数：
        - **ranks** (list[int], 可选) - 设备编号列表。如果为 ``None`` ，创建全局通信组。默认值为 ``None`` 。
        - **timeout** (int, 无效参数) - 当前为预留参数。
        - **backend** (str, 无效参数) - 当前为预留参数。
        - **pg_options** (str, 无效参数) - 当前为预留参数。
        - **use_local_synchronization** (bool, 无效参数) - 当前为预留参数。
        - **group_desc** (str, 无效参数) - 当前为预留参数。

    返回：
        str，生成的通信组名称，如果执行异常则返回空。

    异常：
        - **TypeError** - 在参数 `ranks` 不是列表时或有重复设备号。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
