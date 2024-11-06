# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import subprocess
import json
import socket
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_msrun():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun.
    Expectation: All workers are successfully spawned and running training.
    """
    ms.set_context(jit_level='O0')
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True "\
        "test_msrun.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_exception():
    """
    Feature: 'msrun' launch utility.
    Description: Create python and cpp exception for msrun respectively and check whether cluster could exit
                 and filter out the error logs.
    Expectation: Cluster exits with no process hanging and error log is filtered out.
    """
    # Need to set log level so key words could be filtered out.
    os.environ['GLOG_v'] = str(2)
    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True --log_dir=python_exception_log "\
        "test_msrun_exception.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
        "--exception_type='python'"
    )
    assert result.find("Rank 0 throw python exception.") != -1
    assert result.find("The node: 0 is timed out") != -1


    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True --log_dir=cpp_exception_log "\
        "test_msrun_exception.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
        "--exception_type='cpp'"
    )
    assert result.find("For 'MatMul' the input dimensions must be equal, but got 'x1_col': 84 and 'x2_row': 64") != -1
    assert result.find("The node: 1 is timed out") != -1


def _create_rank_table_file(save_json_to_path, rank_table_dict):
    """
    create rank table file for train or test
    """
    with open(save_json_to_path, "w") as f:
        json.dump(rank_table_dict, f)


def _get_device_ips():
    device_ips = []
    result = subprocess.getoutput("for i in {0..7};do hccn_tool -i $i -ip -g;done")
    lines = result.splitlines()
    for line in lines:
        key, value = line.split(':')
        if key == 'ipaddr':
            device_ips.append(value)
    return device_ips


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_msrun_with_rank_table():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_4p.json", then check whether rank ids are reassigned
                 based on the rank table file and whether initializing hccl comm by rank table is called.
    Expectation: All workers are successfully spawned, their rank ids and device ids are assigned correctly.
    """
    # get a list of real device ips in context.
    device_ips = _get_device_ips()
    # create rank table file for 4 devices with rearranged rank ids.
    rank_table_dict_4p = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{
            "server_id": "10.*.*.*",
            "device": [{"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "3"},
                       {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "2"},
                       {"device_id": "2", "device_ip": "192.3.*.6", "rank_id": "0"},
                       {"device_id": "3", "device_ip": "192.4.*.6", "rank_id": "1"}],
            "host_nic_ip": "reserve",
            "pod_ip": "127.0.0.1"
            }],
        "status": "completed"
        }
    rank_table_dict_4p["server_list"][0]["device"][0]["device_ip"] = device_ips[0]
    rank_table_dict_4p["server_list"][0]["device"][1]["device_ip"] = device_ips[1]
    rank_table_dict_4p["server_list"][0]["device"][2]["device_ip"] = device_ips[2]
    rank_table_dict_4p["server_list"][0]["device"][3]["device_ip"] = device_ips[3]
    _create_rank_table_file("rank_table_4p.json", rank_table_dict_4p)

    ms.set_context(jit_level='O0')
    os.environ['GLOG_v'] = str(1)
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --rank_table_file=rank_table_4p.json --log_dir=./rank_table_reassignment "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert return_code == 0

    result_reassign = subprocess.getoutput("grep -rn ' corresponds to Device_id ' ./rank_table_reassignment")
    assert result_reassign.find("Rank_id [0] corresponds to Device_id [2]") != -1
    assert result_reassign.find("Rank_id [1] corresponds to Device_id [3]") != -1
    assert result_reassign.find("Rank_id [2] corresponds to Device_id [1]") != -1
    assert result_reassign.find("Rank_id [3] corresponds to Device_id [0]") != -1

    result_initialize = subprocess.getoutput("grep -rn 'End to initialize communicator' ./rank_table_reassignment")
    assert result_initialize.find("End to initialize communicator by HcclCommInitClusterInfoConfig for "\
                                  "hccl_world_group") != -1
    assert result_initialize.find("End to initialize communicator by HcclCreateSubCommConfig for hccl_sub_group") != -1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_rank_table_wrong_host_ip():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_4p_wrong_host_ip.json", then check whether CANN error
                 'ranktable invalid' is shown and rank ids are not reassigned.
    Expectation: Log CANN error 'The ranktable or rank is invalid', and rank ids are not be assigned
                 because of wrong HOST_IP.
    """
    # create rank table file for 4 devices with wrong "pod_ip".
    rank_table_dict_4p_wrong_host_ip = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{
            "server_id": "10.*.*.*",
            "device": [{"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "3"},
                       {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "2"},
                       {"device_id": "2", "device_ip": "192.3.*.6", "rank_id": "0"},
                       {"device_id": "3", "device_ip": "192.4.*.6", "rank_id": "1"}],
            "host_nic_ip": "reserve",
            "pod_ip": "reserve"
            }],
        "status": "completed"
        }
    _create_rank_table_file("rank_table_4p_wrong_host_ip.json", rank_table_dict_4p_wrong_host_ip)

    ms.set_context(jit_level='O0')
    os.environ['GLOG_v'] = str(2)
    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --rank_table_file=rank_table_4p_wrong_host_ip.json --log_dir=./rank_table_wrong_host_ip "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert result.find("The ranktable or rank is invalid") != -1
    result_reassign = subprocess.getoutput("grep -rn 't reassign rank id based on rank table file.' "\
                                  "./rank_table_wrong_host_ip/scheduler.log")
    assert result_reassign.find("HOST_IP cannot be found in rank table file") != -1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_rank_table_wrong_device_num():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_dict_4p_wrong_device_num.json", then check whether CANN error
                 'ranktable invalid' is shown and rank ids are not reassigned.
    Expectation: Log CANN error 'The ranktable or rank is invalid', and rank ids are not be assigned
                 because of wrong num of devices.
    """
    # create rank table file for 4 devices with wrong num of device.
    rank_table_dict_4p_wrong_device_num = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{
            "server_id": "10.*.*.*",
            "device": [{"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "3"},
                       {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "2"},
                       {"device_id": "2", "device_ip": "192.3.*.6", "rank_id": "0"}],
            "host_nic_ip": "reserve",
            "pod_ip": "127.0.0.1"
            }],
        "status": "completed"
        }
    _create_rank_table_file("rank_table_dict_4p_wrong_device_num.json", rank_table_dict_4p_wrong_device_num)

    ms.set_context(jit_level='O0')
    os.environ['GLOG_v'] = str(2)
    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 --join=True "\
        "--rank_table_file=rank_table_dict_4p_wrong_device_num.json --log_dir=./rank_table_wrong_device_num "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert result.find("The ranktable or rank is invalid") != -1
    result_reassign = subprocess.getoutput("grep -rn 't reassign rank id based on rank table file.' "\
                                  "./rank_table_wrong_device_num/scheduler.log")
    assert result_reassign.find("is not equal to total number of devices") != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='unessential')
def test_msrun_tail_all_renamed_worker_log():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--tail_worker_log" and "--worker_log_name", then check whether log files are renamed.
    Expectation: All workers are spawned, their log files are successfully renamed.
    """
    ms.set_context(jit_level='O0')
    os.environ['GLOG_v'] = str(2)

    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --log_dir=./tail_all_with_rename --worker_log_name={ip}{hostname} "\
        "test_msrun_only_init.py --device_target=Ascend "\
        "--dataset_path=/home/workspace/mindspore_dataset/mnist > ./all_workers.txt 2>&1"
    )
    hostname = socket.gethostname()
    ip = socket.gethostbyname(socket.gethostname())
    worker_0 = ip + hostname + "_0.log"
    worker_1 = ip + hostname + "_1.log"
    worker_2 = ip + hostname + "_2.log"
    worker_3 = ip + hostname + "_3.log"

    result_tail_all = subprocess.getoutput("grep -rna ' This node ' ./all_workers.txt")
    result_rename_0 = subprocess.getoutput(['find', './tail_all_with_rename', '-name', worker_0])
    result_rename_1 = subprocess.getoutput(['find', './tail_all_with_rename', '-name', worker_1])
    result_rename_2 = subprocess.getoutput(['find', './tail_all_with_rename', '-name', worker_2])
    result_rename_3 = subprocess.getoutput(['find', './tail_all_with_rename', '-name', worker_3])

    assert return_code == 0
    assert result_tail_all.find("This node 0 rank id: 0") != -1
    assert result_tail_all.find("This node 1 rank id: 1") != -1
    assert result_tail_all.find("This node 2 rank id: 2") != -1
    assert result_tail_all.find("This node 3 rank id: 3") != -1
    assert result_rename_0 != -1
    assert result_rename_1 != -1
    assert result_rename_2 != -1
    assert result_rename_3 != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_tail_specified_worker_log():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument "--tail_worker_log"
                 and not set argument "--worker_log_name", then check whether log files are renamed.
    Expectation: All workers are spawned, specified worker log files are successfully output to console.
    """
    ms.set_context(jit_level='O0')
    os.environ['GLOG_v'] = str(2)

    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --tail_worker_log=0,1 --log_dir=./tail_single_without_rename test_msrun_only_init.py "\
        "--device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
        "> ./single_worker.txt 2>&1"
    )
    result_tail_single = subprocess.getoutput("grep -rna ' This node ' ./single_worker.txt")
    result_rename_0 = subprocess.getoutput("find ./tail_all_with_rename -name worker_0.log")
    result_rename_1 = subprocess.getoutput("find ./tail_all_with_rename -name worker_1.log")
    result_rename_2 = subprocess.getoutput("find ./tail_all_with_rename -name worker_2.log")
    result_rename_3 = subprocess.getoutput("find ./tail_all_with_rename -name worker_3.log")

    assert return_code == 0
    assert result_tail_single.find("This node 0 rank id: 0") != -1
    assert result_tail_single.find("This node 1 rank id: 1") != -1
    assert result_tail_single.find("This node 2 rank id: 2") == -1
    assert result_tail_single.find("This node 3 rank id: 3") == -1
    assert result_rename_0 != -1
    assert result_rename_1 != -1
    assert result_rename_2 != -1
    assert result_rename_3 != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_msrun_with_correct_hostname():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with a correct hostname.
    Expectation: Hostname is correctly converted to IP and all workers are successfully spawned.
    """
    os.environ['GLOG_v'] = str(1)
    ms.set_context(jit_level='O0')
    hostname = socket.gethostname()
    ipaddr = socket.gethostbyname(hostname)
    print(f"The hostname of this node is {hostname}, ip address is {ipaddr}.")
    cmd = (f"msrun --worker_num=4 --local_worker_num=4 --master_addr={hostname} "\
            "--master_port=10969 --join=True test_msrun_only_init.py --device_target=Ascend "\
            "--dataset_path=/home/workspace/mindspore_dataset/mnist > ./hostname_normal_msrun.log 2>&1")
    os.system(cmd)
    result = subprocess.getoutput("grep -rn ' to ip address:' ./hostname_normal_msrun.log")
    assert result.find(f"Convert input host name:{hostname} to ip address:{ipaddr}.") != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_wrong_hostname():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with a wrong hostname.
    Expectation: Hostname cannot be converted to IP and a RuntimeError will be raised.
    """
    os.environ['GLOG_v'] = str(2)
    ms.set_context(jit_level='O0')
    hostname = "wrong_hostname"
    print(f"The hostname of this node is {hostname}.")
    cmd = (f"msrun --worker_num=4 --local_worker_num=4 --master_addr={hostname} --master_port=10969 --join=True "\
            "test_msrun_only_init.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
            "> ./hostname_abnormal_msrun.log 2>&1")
    os.system(cmd)
    result = subprocess.getoutput("grep -rn 'DNS resolution failed' ./hostname_abnormal_msrun.log")
    assert result.find("Name or service not known") != -1
