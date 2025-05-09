mindspore.mint.distributed.TCPStore
=====================================

.. py:class:: mindspore.mint.distributed.TCPStore(host_name=None, port=None, world_size=None, is_master=False, timeout=300, wait_for_workers=True, multi_tenant=False, master_listen_fd=None, use_libuv=True)

    一种基于传输控制协议（TCP）的分布式键值存储实现方法。

    .. note::
        - 该功能通过CPU实现，不涉及任何与Ascend相关的硬件操作。
        - 目前TCPStore类构造函数提供的所有参数均不支持。主节点和端口号等都由MindSpore框架统一指定。以下为预留参数，目前不支持，设置无效。
        - 当前TcpStore功能受限，仅支持key小于4k、value小于1G场景，复杂场景待支持。
        - TcpStore功能中消息收发超时时间由 `MS_RECEIVE_MSG_TIMEOUT` 环境变量控制，单位：秒，默认值: ``15`` 。如果出现超时，用户需增加该配置值。

    参数：
        - **host_name** (str，无效参数，可选) - 服务器存储应运行的主机名或 IP 地址。默认值为 ``None`` 。
        - **port** (int，无效参数，可选) - 服务器存储应侦听传入请求的端口。默认值为 ``None`` 。
        - **world_size** (int，无效参数，可选) - 存储用户总数（客户端数量 + 服务器数量 ``1`` ）。默认值为 ``None`` （ ``None`` 表示存储用户数量不固定）。
        - **is_master** (bool，无效参数，可选) - 初始化服务器存储时为 ``True`` ，初始化客户端存储时为 ``False`` 。默认值为 ``False`` 。
        - **timeout** (timedelta，无效参数，可选) - 初始化期间存储使用的超时时间，单位：秒。默认值为 ``300`` 。
        - **wait_for_workers** (bool，无效参数，可选) - 是否等待所有工作进程连接到服务器存储。这仅在 `world_size` 为固定值时适用。默认值为 ``True`` 。
        - **multi_tenant** (bool，无效参数，可选) - 如果为 True，则当前进程中具有相同主机/端口的所有 ``TCPStore`` 实例将使用相同的底层 ``TCPServer`` 。默认值为 ``False`` 。
        - **master_listen_fd** (int，无效参数，可选) - 如果指定，则底层 ``TCPServer`` 将侦听此文件描述符，该文件描述符必须是已绑定到端口的套接字。在某些情况下有助于避免端口分配竞争。默认值为 ``None`` （意味着服务器创建一个新的套接字并尝试将其绑定到端口）。
        - **use_libuv** (bool，无效参数，可选) - 如果为 ``True`` ，则将 libuv 用于 ``TCPServer`` 后端。默认值为 ``True`` 。

    返回：
        ``TCPStore`` 对象。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

    .. py:method:: delete_key(key)

        从存储中删除与 `key` 关联的键值对。

        参数：
            - **key** (str) - 要从存储中删除的 `key` 。

        返回：
            bool，如果成功删除了该键，则返回 ``True`` ，否则返回 ``False`` 。

        异常：
            - **TypeError** - 当 `key` 不是字符串。

        样例：

        .. note::
            .. include:: mindspore.mint.comm_note.rst

    .. py:method:: get(key)

        检索与存储中给定 `key` 关联的值。如果存储中不存在 `key` ，则该函数将返回空字符串。

        参数：
            - **key** (str) - 该函数将返回与此键关联的值。

        返回：
            bytes，如果 `key` 在存储中，则返回与 `key` 关联的值。

        异常：
            - **TypeError** - 当 `key` 不是字符串。

        样例：

        .. note::
            .. include:: mindspore.mint.comm_note.rst

    .. py:method:: set(key, value)

        根据提供的 `key` 和 `value` 将键值对插入到存储中。如果存储中已经存在 `key` ，它将使用新提供的 `value` 覆盖旧值。

        参数：
            - **key** (str) - 要添加到存储中的键。
            - **value** (Union[bytes, str]) - 要添加到存储中的键值对的value。

        异常：
            - **TypeError** - 当 `key` 不是字符串。
            - **TypeError** - 当 `value` 不是字符串或者字节类型。

        样例：

        .. note::
            .. include:: mindspore.mint.comm_note.rst
