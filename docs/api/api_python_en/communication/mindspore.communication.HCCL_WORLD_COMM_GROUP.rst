mindspore.communication.HCCL_WORLD_COMM_GROUP
=============================================

.. py:data:: mindspore.communication.HCCL_WORLD_COMM_GROUP

    The string of "hccl_world_group" referring to the default communication group created by HCCL. On the Ascend hardware platforms, the string is equivalent to ``GlobalComm.WORLD_COMM_GROUP`` after the communication service is initialized. It is recommended to use ``GlobalComm.WORLD_COMM_GROUP`` to obtain the current global communication group.
