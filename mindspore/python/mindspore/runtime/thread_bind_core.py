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

"""Executor manager interfaces."""
import subprocess
from dataclasses import dataclass
from typing import Union
import re
import os
import ast
from mindspore import log as logger
from mindspore import context
from mindspore.communication import get_local_rank_size, get_local_rank


def execute_command(cmd_list):
    try:
        with subprocess.Popen(cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            out, _ = p.communicate(timeout=1000)
        res = out.decode()
        return res
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to execute command, because {e}.")


def _validate_affinity_cpu_list(affinity_cpu_list):
    """
    Validate the user-configured affinity_cpu_list.

    Args:
        affinity_cpu_list (list): Customized bind-core policy to be validated.

    Returns:
        None.
    """
    if isinstance(affinity_cpu_list, dict):
        return False

    if affinity_cpu_list is None:
        return True

    # TODO Remove this check until Mindformers is adapted.
    if not isinstance(affinity_cpu_list, list):
        raise TypeError(f"The parameter '{affinity_cpu_list}' must be list, but got {type(affinity_cpu_list)}")

    range_pattern = re.compile(r'^\d+-\d+$')

    for cpu_range in affinity_cpu_list:
        if not isinstance(cpu_range, str):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be a string.")
        if not range_pattern.match(cpu_range):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be in format 'cpuidX-cpuidY'.")
    return True


def _validate_module_cpu_index(module_to_cpu_dict):
    """
    Validate the user-configured module_to_cpu_dict.

    Args:
        module_to_cpu_dict (dict): Customized module-to-CPU mapping to be validated.

    Returns:
        None.
    """
    if module_to_cpu_dict is None:
        return

    # TODO Remove this check until Mindformers is adapted.
    if not isinstance(module_to_cpu_dict, dict):
        raise TypeError(f"The parameter '{module_to_cpu_dict}' must be dict, but got {type(module_to_cpu_dict)}")

    for module_name, cpu_indices in module_to_cpu_dict.items():
        if not isinstance(cpu_indices, list):
            raise ValueError(f"The value of module_to_cpu_dict: {cpu_indices} should be a list.")
        for cpu_id in cpu_indices:
            if not isinstance(cpu_id, int) or cpu_id < 0:
                raise ValueError(f"CPU index '{cpu_id}' for module '{module_name}' in '{cpu_indices}' "
                                 "should be a non-negative integer.")


def _get_cpu_available():
    """
    Get the CPU resources available on the environment.

    Returns:
        list: List of available CPUs on the environment.
    """
    available_cpu_str = execute_command(["cat", "/sys/fs/cgroup/cpuset/cpuset.cpus"]).strip().split(",")
    available_cpus = list()
    for range_str in available_cpu_str:
        endpoints = range_str.split("-")
        if len(endpoints) != 2:
            raise RuntimeError("'cat /sys/fs/cgroup/cpuset/cpuset.cpus' command output error, please check!")
        available_cpus += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]
    return available_cpus


@dataclass
class DeviceInfo:
    """
    A class to represent information about an Ascend device.

    Attributes:
        _info_line (str): A raw string containing device information.
        npu_id (int): The ID of the NPU.
        chip_id (int): The ID of the chip.
        chip_logic_id (Union[int, str]): The logical ID of the chip, which can be an integer or a string.
        chip_name (str): The name of the chip.

    Methods:
        __post_init__(): Initializes the attributes based on input.
    """
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: Union[int, str] = 0
    chip_name: str = ""

    def __post_init__(self):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = \
            self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


def _get_device_map_info():
    """
    Get abbreviated information about all NPUs on the environment.

    Returns:
        dict: Mapping of NPU logical ID to its details.
        set: Contains all available NPU logical ids on the environment.
    """
    device_map_info = {}
    available_devices = set()
    device_map = \
        execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
    for line in device_map:
        device_info = DeviceInfo(line.strip())
        if isinstance(device_info.chip_logic_id, int):
            device_map_info[device_info.chip_logic_id] = device_info
            available_devices.add(device_info.chip_logic_id)
    return device_map_info, available_devices


def _get_pcie_info(device_map_info, available_devices, keyword="PCIeBusInfo"):
    """
    Get the PCIe number of the NPU device.

    Args:
        device_map_info (dict): A map of NPU logical ID to its details.
        available_devices (set): All available NPU logical ids on the environment.

    Returns:
        dict: Mapping of NPU logical ID to its PCIe number.
    """
    device_pcie_map = {}
    for device in available_devices:
        device_info = device_map_info.get(device)
        if not device_info:
            raise RuntimeError("Can not get device info, binding cpu will skip.")
        pcie_info = \
            execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}",
                             "-c", f"{device_info.chip_id}"]).strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                device_pcie_map[device] = line[len(keyword) + 1:]
                break
    return device_pcie_map


def _get_numa_info(device_pcie_map, keyword="NUMAnode"):
    """
    Get NUNA node affinity for device based on PCIe.

    Args:
        device_pcie_map (dict): A map of NPU logical ID to its PCIe number.

    Returns:
        dict: Mapping of device ID to its affinity NUMA nodes.
        dict: Mapping of NUMA node to its affinity device IDs.
    """
    device_to_numa_map = {}
    numa_to_device_map = {}

    for device, pcie_no in device_pcie_map.items():
        numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).strip().split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_to_numa_map[device] = numa_id

                devices = numa_to_device_map.get(numa_id, None)
                if devices is None:
                    numa_to_device_map[numa_id] = list()
                numa_to_device_map[numa_id].append(device)
                break
    numa_to_device_map[-1] = list(device_pcie_map.keys())
    return device_to_numa_map, numa_to_device_map


def _get_cpu_info(numa_ids, available_cpus, keyword1="NUMAnode", keyword2="CPU(s)"):
    """
    Get information about the CPUs on the NUMA nodes on the environment.

    Args:
        numa_ids (list): A list of NUMA nodes need to get related CPU information.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of NUMA node to its affinity CPUs.
    """
    numa_to_cpu_map = dict()

    cpu_info = execute_command(["lscpu"]).strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if line.startswith(keyword1):
            pattern = re.escape(keyword1) + r'(\d+)' + re.escape(keyword2)
            match = re.search(pattern, line)
            if match:
                numa_id = int(match.group(1))
                split_info = line.split(":")
                cpu_id_ranges = split_info[-1].split(",")
                ranges = list()
                for range_str in cpu_id_ranges:
                    endpoints = range_str.split("-")
                    if len(endpoints) != 2:
                        raise RuntimeError("'lscpu' command output error, please check!")
                    ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1) if cid in available_cpus]
                if numa_id not in numa_ids:
                    numa_id = int(-1)
                if numa_id not in numa_to_cpu_map:
                    numa_to_cpu_map[numa_id] = list()
                numa_to_cpu_map[numa_id].extend(ranges)
    return numa_to_cpu_map


def _get_physical_device_id(logical_device_id):
    """
    Get physical device id from logical device id.

    Args:
        logical_device_id (int): The logical device id for this process in the task.

    Returns:
        int: The physical device id for this process in the host.
    """
    env_visible_device = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    if not env_visible_device:
        physical_device_id = logical_device_id
    else:
        list_visible_device = []
        for item in env_visible_device.split(','):
            list_visible_device.append(int(item))
        list_visible_device.sort()
        physical_device_id = list_visible_device[logical_device_id]
    return physical_device_id


def _equal_distribution_strategy(logical_device_id, device_count, available_cpus):
    """
    Equally distributes available cpus according to logical device id and device count.

    Args:
        logical_device_id (int): The logical device id for this process in the task.
        device_count(int): The total number of device in the task.
        available_cpus(list): A list of cpus in the environment.

    Returns:
        list: A list of cpus assigned to this logical device id.
    """
    physical_device_id = _get_physical_device_id(logical_device_id)
    cpu_num_per_device = int(len(available_cpus)) // device_count
    if cpu_num_per_device < 1:
        logger.warning(f"Available CPUs is less than 1. Will not enable bind core feature.")
        return []
    cpu_start = cpu_num_per_device * physical_device_id
    cpu_end = cpu_start + cpu_num_per_device
    cpu_list_for_device = available_cpus[cpu_start:cpu_end]
    return cpu_list_for_device


def _assemble_env_info(available_devices, available_cpus, affinity_flag, numa_to_cpu_map,
                       device_to_numa_map, logical_device_id):
    """
    Assemble all results of commands based on the hardware on the environment.

    Args:
        available_devices (list): All available NPU logical ids on the environment.
        available_cpus (list): A list of available CPUs on the environment.
        affinity_flag (bool): Whether or not it satisfies generating CPU affinity bind-core policy based on the
          resources on the environment.
        numa_to_cpu_map (dict): A map of NUMA node to its affinity CPUs.
        device_to_numa_map (dict): A map of device ID to its affinity NUMA nodes.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    physical_device_id = _get_physical_device_id(logical_device_id)

    device_to_cpu_map = {}
    for device_id in available_devices:
        device_to_cpu_map[device_id] = list()
    available_cpu_num = len(available_cpus)
    available_device_num = len(available_devices)
    cpu_num_per_device = available_cpu_num // available_device_num
    if cpu_num_per_device < 1:
        logger.warning(f"Available CPUs is less than 1. Will not enable bind core feature.")
        return []

    if affinity_flag:
        device_to_cpu_idx = {}
        for numa_id in numa_to_cpu_map:
            device_to_cpu_idx[numa_id] = 0
        for device_id in available_devices:
            numa_id = device_to_numa_map.get(device_id)
            affinity_cpu_num = 0
            # Prioritize the use of affinity cpu resources.
            affinity_cpu_start_idx = device_to_cpu_idx[numa_id]
            if len(numa_to_cpu_map[numa_id][affinity_cpu_start_idx:]) >= cpu_num_per_device:
                affinity_cpu = numa_to_cpu_map[numa_id][
                    affinity_cpu_start_idx:(affinity_cpu_start_idx + cpu_num_per_device)]
            else:
                affinity_cpu = numa_to_cpu_map[numa_id][affinity_cpu_start_idx:]
            affinity_cpu_num = len(affinity_cpu)
            device_to_cpu_map[device_id].extend(affinity_cpu)
            device_to_cpu_idx[numa_id] = affinity_cpu_start_idx + affinity_cpu_num
            # If the affinity cpu resources are insufficient then use resources from the non-affinity cpu pool.
            if -1 in device_to_cpu_idx:
                unaffinity_cpu_start_idx = device_to_cpu_idx[-1]
                unaffinity_cpu_num = cpu_num_per_device - affinity_cpu_num
                unaffinity_cpu = numa_to_cpu_map[-1][
                    unaffinity_cpu_start_idx:(unaffinity_cpu_start_idx + unaffinity_cpu_num)]
                device_to_cpu_map[device_id].extend(unaffinity_cpu)
                device_to_cpu_idx[-1] = unaffinity_cpu_start_idx + unaffinity_cpu_num
    else:
        device_rank = 0
        for device_id in available_devices:
            cpu_start = device_rank * cpu_num_per_device
            device_to_cpu_map[device_id] = available_cpus[cpu_start:(cpu_start + cpu_num_per_device)]
            device_rank += 1

    cpu_list_for_device = device_to_cpu_map[physical_device_id]
    return cpu_list_for_device


def _auto_generate_policy(logical_device_id, device_count, available_cpus):
    """
    Automatically generate bind-core policy based on CPU affinity.

    Args:
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    device_pcie_map = {}
    device_to_numa_map = {}
    numa_to_device_map = {}
    numa_to_cpu_map = {}
    affinity_flag = False
    # Get the hardware resources in the environment. If this fails, will bind core not based on device.
    try:
        device_map_info, available_devices = _get_device_map_info()
    except RuntimeError as e:
        cpu_list_for_device = _equal_distribution_strategy(logical_device_id, device_count, available_cpus)
        logger.warning(f"Failed to acquire device to numa affinity info, error: {e} "
                       "Will not bind core based on affinity.")
        return cpu_list_for_device
    # Get the affinity resources in the environment. If this fails, will bind core not based on affinity.
    try:
        device_pcie_map = _get_pcie_info(device_map_info, available_devices)
        device_to_numa_map, numa_to_device_map = _get_numa_info(device_pcie_map)
        numa_to_cpu_map = _get_cpu_info(list(numa_to_device_map.keys()), available_cpus)
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device to numa affinity info, error: {e} "
                       "Will not bind core based on affinity.")
        affinity_flag = False
    if device_pcie_map and device_to_numa_map and numa_to_device_map and numa_to_cpu_map:
        affinity_flag = True
    # Auto-generation of bind core policy for Ascned.
    try:
        cpu_list_for_device = _assemble_env_info(available_devices, available_cpus, affinity_flag,
                                                 numa_to_cpu_map, device_to_numa_map, logical_device_id)
        return cpu_list_for_device
    except (RuntimeError, ZeroDivisionError) as e:
        logger.warning(f"Failed to auto generate bind core policy, error: {e}. "
                       "Will not enable bind core feature.")
        return []


def _customize_generate_policy(affinity_cpu_list, available_cpus):
    """
    Generate customized bind-core policy based on user-configured inputs.

    Args:
        affinity_cpu_list (list): User-configured inputs to generate customized bind-core policy.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    cpu_list_for_device = list()

    for cpu_range_str in affinity_cpu_list:
        endpoints = cpu_range_str.split("-")
        for cid in range(int(endpoints[0]), int(endpoints[1]) + 1):
            if cid not in available_cpus:
                raise RuntimeError(f"CPU id:{cid} set in affinity_cpu_list:{affinity_cpu_list} is not available.")
            cpu_list_for_device.append(cid)

    if not cpu_list_for_device:
        logger.warning(f"Available CPUs is less than 1. Will not enable bind core feature.")

    return cpu_list_for_device


def _assign_cpu_to_module(cpu_list_for_device, module_to_cpu_dict):
    """
    Assign specific CPUs to modules.

    Args:
        cpu_list_for_device (list): A map of device to its affinity CPUs.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
    """
    module_bind_core_policy = {}

    valid_module_names = {"main", "runtime", "pynative", "minddata"}

    if module_to_cpu_dict is not None:
        module_bind_core_policy = {
            module: [cpu_list_for_device[i] for i in indices if 0 <= i < len(cpu_list_for_device)]
            for module, indices in module_to_cpu_dict.items() if module in valid_module_names
        }
    else:
        module_bind_core_policy["main"] = cpu_list_for_device
    return module_bind_core_policy


def _get_cpu_affinity_policy(affinity_cpu_list=None, module_to_cpu_dict=None):
    """
    The entry to get bind-core policy.

    Args:
        affinity_cpu_list (list, optional): User-configured inputs to generate customized bind-core policy.
          Default: ``None``.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
        bool: Whether the generated bind-core policy is based on cpu affinity.
    """
    device_target = context.get_context("device_target")

    # Get the CPU resources in the environment. If this fails, the binding core feature will not be enabled.
    try:
        available_cpus = _get_cpu_available()
    except RuntimeError as e:
        logger.warning(f"Failed to acquire available cpu info, error: {e} Will not enable bind core feature.")
        return {}
    if (affinity_cpu_list is not None) and (affinity_cpu_list):
        # User configured binding core policy.
        cpu_list_for_device = _customize_generate_policy(affinity_cpu_list, available_cpus)
    else:
        # Automatic generation of binding core policy based on resources on the environment.
        env_msrun_cpu_list = os.getenv("MSRUN_CPU_LIST")
        if env_msrun_cpu_list:
            module_bind_core_policy = _assign_cpu_to_module(ast.literal_eval(env_msrun_cpu_list), module_to_cpu_dict)
            logger.warning(f"Module bind core policy from msrun: {module_bind_core_policy}.")
            return module_bind_core_policy
        try:
            logical_device_id = get_local_rank()
            device_count = get_local_rank_size()
        except RuntimeError as e:
            logger.warning(f"Fail to get device_id or device_count, error: {e} Will not enable bind core feature.")
            return {}
        # If the device target is Ascend, the affinity between the device and NUMA node is taken into account
        # to generate the binding core policy.
        if device_target == "Ascend":
            cpu_list_for_device = _auto_generate_policy(logical_device_id, device_count, available_cpus)
        else:
            cpu_list_for_device = _equal_distribution_strategy(logical_device_id, device_count, available_cpus)
    # cpu_list_for_device is empty, indicating that the basic conditions have not been met to enable the thread bind core feature.
    if not cpu_list_for_device:
        return {}
    module_bind_core_policy = _assign_cpu_to_module(cpu_list_for_device, module_to_cpu_dict)
    logger.warning(f"Module bind core policy generated: {module_bind_core_policy}.")
    return module_bind_core_policy
