mindspore.communication.MCCL_WORLD_COMM_GROUP
=============================================

.. py:data:: mindspore.communication.MCCL_WORLD_COMM_GROUP

    The string of "mccl_world_group" referring to the default communication group created by MCCL. On the CPU hardware platforms, the string is equivalent to ``GlobalComm.WORLD_COMM_GROUP`` after the communication service is initialized. It is recommended to use ``GlobalComm.WORLD_COMM_GROUP`` to obtain the current global communication group.
