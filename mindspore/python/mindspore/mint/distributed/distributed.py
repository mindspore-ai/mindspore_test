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
import os
from mindspore import context
from mindspore import log as logger
from mindspore.communication._comm_helper import Backend, _get_rank_helper, _get_size_helper, \
    _create_group_helper, _destroy_group_helper, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    MCCL_WORLD_COMM_GROUP, DEVICE_TO_BACKEND, _get_local_rank_helper, _get_local_size_helper, GlobalComm
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.tensor import Tensor
from mindspore.communication import init, release



def init_process_group(
    backend = "hccl",
    init_method = None,
    timeout = None,
    world_size = -1,
    rank = -1,
    store = None,
    pg_option = None,
    device_id = None
):
    """
    Init collective communication lib. And create a default collective communication group.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.

    Args:
        backend (str, optional): The backend to ues. default is hccl and now only support hccl.
        init_method (str, invalid): URL specifying how to init collective communication group. Provides paramters consistent with pytorch, but is not currently support, setting is invalid.
        tineout (timedelta, invalid): Timeout for API executed. Provides paramters consistent with pytorch, but is not currently support, setting is invalid.
        world_size (int, optional): Number of the processes participating in the job.
        rank (int, invalid): Rank of the current process. Provides paramters consistent with pytorch, but is not currently support, setting is invalid.
        store (Store, invalid): Key/Value store accessible to all workers, used to exchange connection/address infomation. Provides paramters consistent with pytorch, but is not currently support, setting is invalid.
        pg_option (ProcessGroupOptions, invalid): process group options specifying what additional options need to be passed in during the construction of specific process group. rovides paramters consistent with pytorch, but is not currently support, setting is invalid.
        device_id (int, invalid): the device id to exeute. rovides paramters consistent with pytorch, but is not currently support, setting is invalid. 

    Raises:
        ValueError: If `backend` is not hccl.
        ValueError: If `world_size` is not equal to -1 or process group number.
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
        >>> from mindspore import ops
        >>> from mindspore.distributed import init_process_group, destroy_process_group
        >>> set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        >>> init_process_group() 
        >>> destroy_process_group()
    """
    if init_method != None:
        logging.warning("init_method is ignore, setting is invalid")
    if timeout != None:
        logging.warning("timeout is ignore, setting is invalid")
    if store != None:
        logging.warning("store is ignore, setting is invalid")
    if pg_option != None:
        logging.warning("pg_option is ignore, setting is invalid")
    if device_id != None:
        logging.warning("device_id is ignore, setting is invalid")
    if backend != "hccl":
        raise ValueError("Only support hccl now, please setting backend to hccl or using default value")

    #init hccl
    init("hccl")

    #create default group
    group_name  = GlobalComm.WORLD_COMM_GROUP
    if world_size = -1:
        world_size = get_world_size()
    else:
        if world_size != get_world_size()
            raise ValueError("world_size is wrong, please using default value or setting: ", get_world_size())
    rank_ids = list(range(0, world_size, 1))
    _create_group_helper(GlobalComm.WORLD_COMM_GROUP, rank_ids)

def destroy_process_group(group = None):
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
    if not isinstance(group, str):
        raise TypeError("For 'destroy_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    if group == "hccl_world_group" or group == None:
        release()
    else:
        _destroy_group_helper(group)
