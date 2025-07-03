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
import pytest
from mindspore import context
from mindspore.mint.distributed.distributed import (
    init_process_group,
    barrier,
    get_rank,
    TCPStore,
    get_world_size
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_tcp_store.py
init_process_group()
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
this_rank = get_rank()
size = get_world_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
def test_TCPStore():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    TCPStore()
    TCPStore("11")


def test_TCPStore_TypeError():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    store = TCPStore("")
    with pytest.raises(TypeError):
        store.set("key1", 1)
    with pytest.raises(TypeError):
        store.set(2, "{'a':1}")
    with pytest.raises(TypeError):
        store.set("key3", [1, 2, 3])
    with pytest.raises(TypeError):
        store.set("key5")
    with pytest.raises(TypeError):
        store.delete_key(4)
    with pytest.raises(TypeError):
        store.get(2)


def test_set():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    store = TCPStore()
    if this_rank == 0:
        store.set("first_key", "value1")
    barrier()
    if this_rank == 1:
        store.set("first_key", "value2")
    barrier()
    data = store.get("first_key")
    assert data.decode() == "value2"


def test_get():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    store = TCPStore()
    if this_rank == 0:
        data = store.get("second_key")
        assert data.decode() == ""
        store.set("second_key", "value")
    barrier()
    if this_rank == 1:
        data = store.get("second_key")
        assert data.decode() == "value"
    barrier()


def test_get_1G():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    store = TCPStore()
    if this_rank == 1:
        data = store.get("third_key")
        assert data.decode() == ""
        value = 'A' * 1024 * 1024 * 1024
        store.set("third_key", value)
    barrier()
    data = store.get("third_key")
    assert len(data) == 1024 * 1024 * 1024
    assert data[0] == 65
    barrier()


def test_delete():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    store = TCPStore()
    if this_rank == 0:
        store.set("first_key", "value")
        ret = store.delete_key("first_key")
        assert ret is True
        ret = store.delete_key("first_key")
        assert ret is False
        data = store.get("first_key")
        assert data.decode() == ""
    barrier()
    if this_rank == 1:
        data = store.get("first_key")
        assert data.decode() == ""
    barrier()
