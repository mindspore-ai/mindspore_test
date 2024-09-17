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
from mindspore.communication._comm_helper import _destroy_group_helper, GlobalComm
from mindspore.communication import init, release, get_group_size

def init_process_group(backend="hccl",
                       init_method=None,
                       timeout=None,
                       world_size=-1,
                       rank=-1,
                       store=None,
                       pg_option=None,
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
        tineout (timedelta, invalid): Timeout for API executed. Provides parameters consistent with pytorch, but is not
                                      currently support, setting is invalid.
        world_size (int, optional): Number of the processes participating in the job.
        rank (int, invalid): Rank of the current process. Provides parameters consistent with pytorch, but is not
                             currently support, setting is invalid.
        store (Store, invalid): Key/Value store accessible to all workers, used to exchange connection/address
                                information. Provides parameters consistent with pytorch, but is not currently support,
                                setting is invalid.
        pg_option (ProcessGroupOptions, invalid): process group options specifying what additional options need to be
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
        >>> from mindspore import ops
        >>> from mindspore.distributed import init_process_group, destroy_process_group
        >>> set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        >>> init_process_group()
        >>> destroy_process_group()
    """
    if init_method is not None:
        logger.warning("init_method is ignored, setting is invalid")
    if timeout is not None:
        logger.warning("timeout is ignored, setting is invalid")
    if store is not None:
        logger.warning("store is ignored, setting is invalid")
    if pg_option is not None:
        logger.warning("pg_option is ignored, setting is invalid")
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
        >>> from mindspore.distributed import init_process_group, destroy_process_group
        >>> set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        >>> init_process_group()
        >>> destroy_process_group()
    """

    if group == GlobalComm.WORLD_COMM_GROUP or group is None:
        release()
    elif not isinstance(group, str):
        raise TypeError("For 'destroy_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    else:
        _destroy_group_helper(group)
