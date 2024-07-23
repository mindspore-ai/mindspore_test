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
import socket
import json
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


def create_rank_table_file(save_json_to_path, rank_table_dict):
    """
    create rank table file for train or test
    """
    with open(save_json_to_path, "w") as f:
        json.dump(rank_table_dict, f)

# create rank table file for 4 devices with rearranged rank ids.
host_ip = socket.gethostbyname(socket.gethostname())
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
        "pod_ip": "reserve"
        }],
    "status": "completed"
    }
rank_table_dict_4p["server_list"][0]["pod_ip"] = host_ip
create_rank_table_file("rank_table_4p.json", rank_table_dict_4p)

# create rank table file for 4 devices with wrong host nic ip.
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
create_rank_table_file("rank_table_4p_wrong_host_ip.json", rank_table_dict_4p_wrong_host_ip)

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
        "pod_ip": "reserve"
        }],
    "status": "completed"
    }
rank_table_dict_4p_wrong_device_num["server_list"][0]["pod_ip"] = host_ip
create_rank_table_file("rank_table_dict_4p_wrong_device_num.json", rank_table_dict_4p_wrong_device_num)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_rank_table():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_4p.json", then check whether rank ids are reassigned
                 based on the rank table file.
    Expectation: All workers are successfully spawned, their rank ids and device ids are assigned correctly.
    """
    ms.set_context(jit_level='O0')
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --rank_table_file=rank_table_4p.json --log_dir=./rank_table_reassignment "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    result = subprocess.getoutput("grep -rn ' corresponds to Device_id ' ./rank_table_reassignment")
    assert return_code == 0
    assert result.find("Rank_id [0] corresponds to Device_id [2]") != -1
    assert result.find("Rank_id [1] corresponds to Device_id [3]") != -1
    assert result.find("Rank_id [2] corresponds to Device_id [1]") != -1
    assert result.find("Rank_id [3] corresponds to Device_id [0]") != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_rank_table_wrong_host_ip():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_4p_wrong_host_ip.json", then check whether rank ids
                 are not reassigned.
    Expectation: All workers are successfully spawned, their rank ids and device ids are assigned in order.
    """
    ms.set_context(jit_level='O0')
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 "\
        "--join=True --rank_table_file=rank_table_4p_wrong_host_ip.json --log_dir=./rank_table_wrong_host_ip "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    result = subprocess.getoutput("grep -rn ' corresponds to Device_id ' ./rank_table_wrong_host_ip")
    assert return_code == 0
    assert result.find("Rank_id [0] corresponds to Device_id [0]") != -1
    assert result.find("Rank_id [1] corresponds to Device_id [1]") != -1
    assert result.find("Rank_id [2] corresponds to Device_id [2]") != -1
    assert result.find("Rank_id [3] corresponds to Device_id [3]") != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_msrun_with_rank_table_wrong_device_num():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun with argument
                 "--rank_table_file rank_table_dict_4p_wrong_device_num.json", then check whether rank ids
                 are not reassigned.
    Expectation: All workers are successfully spawned, their rank ids and device ids are assigned in order.
    """
    ms.set_context(jit_level='O0')
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10969 --join=True "\
        "--rank_table_file=rank_table_dict_4p_wrong_device_num.json --log_dir=./rank_table_wrong_device_num "\
        "test_msrun_rank_table.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    result = subprocess.getoutput("grep -rn ' corresponds to Device_id ' ./rank_table_wrong_device_num")
    assert return_code == 0
    assert result.find("Rank_id [0] corresponds to Device_id [0]") != -1
    assert result.find("Rank_id [1] corresponds to Device_id [1]") != -1
    assert result.find("Rank_id [2] corresponds to Device_id [2]") != -1
    assert result.find("Rank_id [3] corresponds to Device_id [3]") != -1
