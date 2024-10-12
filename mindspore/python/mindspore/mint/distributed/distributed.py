# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Communication management API"""
from mindspore import log as logger
from mindspore.communication._comm_helper import _destroy_group_helper, GlobalComm, _get_rank_helper, _get_size_helper
from mindspore.communication import init, release, get_group_size


def init_process_group(backend="hccl",
                       init_method=None,
                       timeout=None,
                       world_size=-1,
                       rank=-1,
                       store=None,
                       pg_options=None,
                       device_id=None):
    """
    Init collective communication lib. And create a default collective communication group.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.
        In Ascend hardware platforms, this API should be set before the definition of any Tensor and Parameter,
        and the instantiation and execution of any operation and net.

    Args:
        backend (str, optional): The backend to ues. default is hccl and now only support hccl.
        init_method (str, invalid): URL specifying how to init collective communication group. Provides parameters
                                    consistent with pytorch, but is not currently support, setting is invalid.
        timeout (timedelta, invalid): Timeout for API executed. Provides parameters consistent with pytorch, but is not
                                      currently support, setting is invalid.
        world_size (int, optional): Number of the processes participating in the job.
        rank (int, invalid): Rank of the current process. Provides parameters consistent with pytorch, but is not
                             currently support, setting is invalid.
        store (Store, invalid): Key/Value store accessible to all workers, used to exchange connection/address
                                information. Provides parameters consistent with pytorch, but is not currently support,
                                setting is invalid.
        pg_options (ProcessGroupOptions, invalid): process group options specifying what additional options need to be
                                                  passed in during the construction of specific process group. Provides
                                                  parameters consistent with pytorch, but is not currently support,
                                                  setting is invalid.
        device_id (int, invalid): the device id to exeute. Provides parameters consistent with pytorch, but is not
                                  currently support, setting is invalid.

    Raises:
        ValueError: If `backend` is not hccl.
        ValueError: If `world_size` is not equal to -1 or process group number.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails,
                      or the environment variables RANK_ID/MINDSPORE_HCCL_CONFIG_PATH
                      have not been exported when backend is HCCL.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore.mint.distributed import init_process_group, destroy_process_group
        >>> set_context(device_target="Ascend")
        >>> init_process_group()
        >>> destroy_process_group()
    """
    if init_method is not None:
        logger.warning("init_method is ignored, setting is invalid")
    if timeout is not None:
        logger.warning("timeout is ignored, setting is invalid")
    if store is not None:
        logger.warning("store is ignored, setting is invalid")
    if pg_options is not None:
        logger.warning("pg_options is ignored, setting is invalid")
    if device_id is not None:
        logger.warning("device_id is ignored, setting is invalid")
    if rank != -1:
        logger.warning("rank is ignored, setting is invalid")
    if backend != "hccl":
        raise ValueError("Only support hccl now, please setting backend to hccl or using default value")

    #init hccl & create world group
    init(backend)

    if world_size != -1 and world_size != get_group_size():
        raise ValueError("world_size is wrong, please using default value or setting: ", get_group_size())


def destroy_process_group(group=None):
    """
    Destroy the user collective communication group.
    If group is None or "hccl_world_group", Destroy all group and release collective communication lib.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.
        This method should be used after init_process_group().

    Args:
        group (str): The communication group to destroy, the group should be created by init_process_group or new_group.

    Raises:
        TypeError: If group is not a string.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore.mint.distributed import init_process_group, destroy_process_group
        >>> set_context(device_target="Ascend")
        >>> init_process_group()
        >>> destroy_process_group()
    """

    if group == GlobalComm.WORLD_COMM_GROUP or group is None:
        release()
    elif not isinstance(group, str):
        raise TypeError("For 'destroy_group', the argument 'group' must be type of string or None, "
                        "but got 'group' type : {}.".format(type(group)))
    else:
        _destroy_group_helper(group)


def get_rank(group=None):
    """
    Get the rank ID for the current device in the specified collective communication group.

    Note:
        This method should be used after init().

    Args:
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. If None, ``GlobalComm.WORLD_COMM_GROUP`` will be used.

    Returns:
        int, the rank ID of the calling process within the group.
        return -1, if not part of the group

    Raises:
        TypeError: If group is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> from mindspore import set_context
        >>> from mindspore.mint.distributed import init_process_group, get_rank
        >>> set_context(device_target="Ascend")
        >>> init_process_group()
        >>> rank_id = get_rank()
        >>> print(rank_id)
        >>> # the result is the rank_id in world_group
    """
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError("For 'get_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    try:
        ret = _get_rank_helper(group)
    except RuntimeError as e:
        logger.warning(e)
        ret = -1
    return ret


def get_world_size(group=None):
    """
    Get the rank size of the specified collective communication group.

    Note:
        This method should be used after init().

    Args:
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. If None, ``GlobalComm.WORLD_COMM_GROUP`` will be used.

    Returns:
        int, the rank size of the group.
        return -1, if the group is not available.

    Raises:
        TypeError: If group is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore.mint.distributed import init_process_group, get_world_size
        >>> set_context(device_target="Ascend")
        >>> init_process_group()
        >>> group_size = get_world_size()
        >>> print("group_size is: ", group_size)
        group_size is: 8
    """
    ret = -1
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError("For 'get_group_size', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    try:
        ret = _get_size_helper(group)
    except RuntimeError as e:
        logger.warning(e)
        ret = -1
    return ret
