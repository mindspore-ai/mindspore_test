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
from mindspore import log as logger
from mindspore import context


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
        affinity_cpu_list (dict): Customized bind-core policy to be validated.

    Returns:
        None.
    """
    device_pattern = re.compile(r'^device\d+$')
    range_pattern = re.compile(r'^\d+-\d+$')

    for key, value in affinity_cpu_list.items():
        if not isinstance(key, str):
            raise ValueError(f"The key of affinity_cpu_list: {key} should be a string.")
        if not device_pattern.match(key):
            raise ValueError(f"The key of affinity_cpu_list: {key} should be in format 'deviceX'.")
        if not isinstance(value, list):
            raise ValueError(f"The value of affinity_cpu_list: {value} should be a list.")
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"The value of affinity_cpu_list: {item} should be a string.")
            if not range_pattern.match(item):
                raise ValueError(f"The value of affinity_cpu_list: {item} should be in format 'cpuidX-cpuidY'.")


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


def _auto_generate_policy(available_devices, available_cpus, affinity_flag, numa_to_cpu_map, device_to_numa_map):
    """
    Automatically generate bind-core policy based on CPU affinity.

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
    device_to_cpu_map = {}
    for device_id in available_devices:
        device_to_cpu_map[device_id] = list()

    available_cpu_num = len(available_cpus)
    available_device_num = len(available_devices)
    cpu_num_per_device = available_cpu_num // available_device_num
    if cpu_num_per_device < 7:
        raise RuntimeError(f"Cpu num available for each device is {cpu_num_per_device}, "
                           "which is less than the minimum cpu num need. Will not enable bind core feature.")

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
    return device_to_cpu_map


def _customize_generate_policy(affinity_cpu_list, available_cpus):
    """
    Generate customized bind-core policy based on user-configured inputs.

    Args:
        affinity_cpu_list (dict): User-configured inputs to generate customized bind-core policy.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    device_to_cpu_map = {}
    _validate_affinity_cpu_list(affinity_cpu_list)
    for device, cpu_id_ranges in affinity_cpu_list.items():
        ranges = list()
        for range_str in cpu_id_ranges:
            endpoints = range_str.split("-")
            for cid in range(int(endpoints[0]), int(endpoints[1]) + 1):
                if cid not in available_cpus:
                    raise RuntimeError(f"CPU id: {cid} set in affinity_cpu_list is not available.")
                ranges.append(cid)
        if len(ranges) < 7:
            raise RuntimeError(f"cpu num available for {device} is less than 7, which is the minimum cpu num need.")
        device_id = int(device.replace("device", ""))
        device_to_cpu_map[device_id] = ranges
    return device_to_cpu_map


def _assign_cpu_to_module(device_to_cpu_map):
    """
    Assign specific CPUs to modules.

    Args:
        device_to_cpu_map (dict): A map of device to its affinity CPUs.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
    """
    module_bind_core_policy = {}
    for device, cpu_list in device_to_cpu_map.items():
        thread_to_cpu_map = {}
        thread_to_cpu_map["main"] = [cpu_list[0]]
        thread_to_cpu_map["runtime"] = cpu_list[1:6]
        thread_to_cpu_map["pynative"] = cpu_list[1:5]
        thread_to_cpu_map["minddata"] = cpu_list[6:]
        module_bind_core_policy[device] = thread_to_cpu_map
    return module_bind_core_policy


def _get_cpu_affinity_policy(affinity_cpu_list=None):
    """
    The entry to get bind-core policy.

    Args:
        affinity_cpu_list (dict, optional): User-configured inputs to generate customized bind-core policy.
          Default: ``None``.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
        bool: Whether the generated bind-core policy is based on cpu affinity.
    """
    device_target = context.get_context("device_target")
    device_pcie_map = {}
    device_to_numa_map = {}
    numa_to_device_map = {}
    numa_to_cpu_map = {}
    affinity_flag = False
    bind_policy_flag = False

    # Get the CPU resources in the environment. If this fails, the binding core feature will not be enabled.
    try:
        available_cpus = _get_cpu_available()
    except RuntimeError as e:
        logger.warning(f"Failed to acquire available cpu info, error: {e} Will not enable bind core feature.")
        return {}, False
    # Automatic generation of binding core policy based on resources on the environment.
    if (affinity_cpu_list is None) or (not affinity_cpu_list):
        # If the device target is Ascend, the affinity between the device and NUMA node is taken into account
        # to generate the binding core policy.
        if device_target == "Ascend":
            # Get the hardware resources in the environment. If this fails, will bind core not based on device.
            try:
                device_map_info, available_devices = _get_device_map_info()
            except RuntimeError as e:
                logger.warning(f"Failed to acquire device to numa affinity info, error: {e} "
                               "Will not bind core based on affinity. Module bind core policy "
                               f"generated: {available_cpus}.")
                return available_cpus, bind_policy_flag
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
                device_to_cpu_map = _auto_generate_policy(available_devices, available_cpus, affinity_flag,
                                                          numa_to_cpu_map, device_to_numa_map)
            except (RuntimeError, ZeroDivisionError) as e:
                logger.warning(f"Failed to auto generate bind core policy, error: {e}. "
                               "Will not enable bind core feature.")
                return {}, False
            module_bind_core_policy = _assign_cpu_to_module(device_to_cpu_map)
            bind_policy_flag = True
        else:
            module_bind_core_policy = available_cpus
    # User configured binding core policy.
    else:
        device_to_cpu_map = _customize_generate_policy(affinity_cpu_list, available_cpus)
        module_bind_core_policy = _assign_cpu_to_module(device_to_cpu_map)
        bind_policy_flag = True
    logger.warning(f"Module bind core policy generated: {module_bind_core_policy}.")
    return module_bind_core_policy, bind_policy_flag
