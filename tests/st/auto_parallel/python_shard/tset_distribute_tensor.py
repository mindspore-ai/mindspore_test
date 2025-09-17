import os
from tests.mark_utils import arg_mark


def run_case(case_name, master_port):
    cmd = f"export GLOG_v=3 && msrun --worker_num=8 --local_worker_num=8 " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"--join=True --log_dir=./{case_name} pytest -s -v " \
          f"distribute_tensor.py::{case_name}"
    ret = os.system(cmd)
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_distribute_on_1dmesh():
    '''
    Feature: Test shard and replicate with 1d mesh.
    Description: Test distribute_tensor()
    Expectation: Run success.
    '''
    case_name = "test_distribute_on_1dmesh"
    master_port = 116677
    run_case(case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_distribute_on_2dmesh():
    '''
    Feature: Test device_matrix=(2, 4), alias=("replicate", "dp"), layout=("dp", None)
    Description: Test distribute_tensor()
    Expectation: Run success.
    '''
    case_name = "test_distribute_on_2dmesh"
    master_port = 116677
    run_case(case_name, master_port)
