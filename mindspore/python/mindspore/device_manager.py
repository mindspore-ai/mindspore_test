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

"""Device manager interfaces."""

import os
from mindspore import log as logger
from mindspore._c_expression import DeviceManagerConf, DeviceContextManager
from mindspore._checkparam import args_type_check
from mindspore.parallel._ps_context import _need_reset_device_target_for_ps

__all__ = ['set_device', 'set_deterministic']

@args_type_check(device_target=str, device_id=int)
def set_device(device_target, device_id=None):
    """
    Set device target and device id for running environment.

    Note:
        - The `device_target` must be set in the ["CPU", "GPU", "Ascend"], there is no default value.

    Args:
        device_target (str): The target device to run, only support "Ascend", "GPU", and "CPU".
        device_id (int): ID of the target device, the value must be in [0, device_num_per_host-1], Default: ``0`` .

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
    """
    valid_targets = ["CPU", "GPU", "Ascend"]
    if device_target not in valid_targets:
        raise ValueError(f"The argument 'device_target' must be one of {valid_targets}, but got {device_target}.")
    # If in Parameter Server mode, Ascend card should not be used by server and scheduler.
    if _need_reset_device_target_for_ps(device_target):
        logger.info("Reset device target to CPU when set_device.")
        device_target = "CPU"

    if DeviceManagerConf.get_instance().is_device_enable():
        raise RuntimeError("The 'mindspore.set_device' can not be set repeatedly.")

    device_context = DeviceContextManager.get_instance().get_device_context(device_target)
    if device_context is not None and device_context.initialized():
        raise RuntimeError("The runtime has been initialized, please set it before the kernel is executed."
                           "Suggest setting it as early as possible.")

    if device_id is None:
        DeviceManagerConf.get_instance().set_device(device_target, 0, True)
    else:
        if device_id < 0:
            raise ValueError("The device id must bigger than or equal to 0.")
        DeviceManagerConf.get_instance().set_device(device_target, device_id, False)

@args_type_check(deterministic=bool)
def set_deterministic(deterministic):
    """
    Whether to enable op run in deterministic mode.

    Args:
        deterministic (bool): Whether to enable op run in deterministic mode, Default:  ``false``.
    Examples:
        >>> import mindspore as ms
        >>> ms.set_deterministic(True)
    """
    # Check the configuration environment whether valid.
    _check_runtime_conf_env_valid()
    if DeviceManagerConf.get_instance().is_deterministic_configured():
        raise RuntimeError("The 'mindspore.set_deterministic' can not be set repeatedly.")

    # Check the hccl_deterministic and te_parallel_compiler.
    hccl_deterministic = os.getenv("HCCL_DETERMINISTIC")
    te_parallel_compiler = os.getenv("TE_PARALLEL_COMPILER")
    if deterministic:
        if hccl_deterministic and hccl_deterministic != "true":
            logger.warning(f"Environment 'HCCL_DETERMINISTIC' should be 'true' when set deterministic='True', but "
                           f"got '{hccl_deterministic}'. 'HCCL_DETERMINISTIC' will be set to 'true'.")
        if te_parallel_compiler and te_parallel_compiler != "1":
            logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should be '1' when set deterministic='True', but "
                           f"got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' will be set to '1'.")
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["TE_PARALLEL_COMPILER"] = "1"
    else:
        if hccl_deterministic and hccl_deterministic != "false":
            logger.warning(f"Environment 'HCCL_DETERMINISTIC' should not be set or be 'false' when set "
                           f"deterministic='False', but got '{hccl_deterministic}'. 'HCCL_DETERMINISTIC' "
                           f"will be unset.")
            del os.environ["HCCL_DETERMINISTIC"]
        if te_parallel_compiler and te_parallel_compiler != "0":
            logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should not be set or be '0' when set "
                           f"deterministic='False', but got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' "
                           f"will be unset.")
            del os.environ["TE_PARALLEL_COMPILER"]

    DeviceManagerConf.get_instance().set_deterministic(deterministic)

def _check_runtime_conf_env_valid():
    """
    Check whether the configuration environment of runtime is valid. If the environment is invalid, throw an exception.
    The two checks are as follows:
    1. Check the device must be initialized:
    The 'mindspore.set_device' is configured to indicate that the device has been initialized.
    2. Check the runtime cannot be initialized:
    The kernel on the device has been executed to indicate that the runtime has been initialized.
    """
    if not DeviceManagerConf.get_instance().is_device_enable():
        raise RuntimeError("The device has not been initialized, please set 'mindspore.set_device' first.")

    device_target = DeviceManagerConf.get_instance().get_device_target()
    device_context = DeviceContextManager.get_instance().get_device_context(device_target)
    if device_context is not None and device_context.initialized():
        raise RuntimeError("The runtime has been initialized, please set it before the kernel is executed."
                           "Suggest setting it as early as possible, but after the 'mindspore.set_device'.")
