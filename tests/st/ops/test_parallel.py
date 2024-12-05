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

import os

from tests import mark_utils

WORKER_NUM = 8


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_all_gather_matmul_normal() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 mindspore.ops.AllGahter and mindspore.ops.MatMul forword calculation.
    """
    status = os.system(f'''
        msrun \
            --worker_num {WORKER_NUM} \
            --local_worker_num {WORKER_NUM} \
            --join True \
            --master_port 6221 \
            --log_dir test_all_gather_matmul_normal \
            pytest -vra parallel/all_gather_matmul.py::test_all_gather_matmul_normal
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_all_gather_matmul_normal_disable_kernel_backoff() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation disabling kernel backoff.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 mindspore.ops.AllGahter and mindspore.ops.MatMul forword calculation.
    """
    status = os.system(f'''
        export MS_DISABLE_KERNEL_BACKOFF=1 \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_all_gather_matmul_normal_disable_kernel_backoff \
               pytest -vra parallel/all_gather_matmul.py::test_all_gather_matmul_normal
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_all_gather_matmul_dynamic() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of forword
                 calculation with inputs in static shapes.
    """
    status = os.system(f'''
        msrun \
            --worker_num {WORKER_NUM} \
            --local_worker_num {WORKER_NUM} \
            --join True \
            --master_port 6221 \
            --log_dir test_all_gather_matmul_dynamic \
            pytest -vra parallel/all_gather_matmul.py::test_all_gather_matmul_dynamic
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_all_gather_matmul_binary_cases() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 torch_npu.npu_all_gather_base_mm forword calculation.
    """
    status = os.system(f'''
        msrun \
            --worker_num {WORKER_NUM} \
            --local_worker_num {WORKER_NUM} \
            --join True \
            --master_port 6221 \
            --log_dir test_all_gather_matmul_binary_cases \
            pytest -vra parallel/all_gather_matmul.py::test_all_gather_matmul_binary_cases
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_all_gather_matmul_binary_cases_disable_kernel_backoff() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation disabling kernel backoff.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 torch_npu.npu_all_gather_base_mm forword calculation.
    """
    status = os.system(f'''
        export MS_DISABLE_KERNEL_BACKOFF=1 \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_all_gather_matmul_binary_cases_disable_kernel_backoff \
               pytest -vra parallel/all_gather_matmul.py::test_all_gather_matmul_binary_cases
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_matmul_reduce_scatter_normal() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
                 mindspore.ops.MatMul and mindspore.ops.ReduceScatter forword calculation.
    """
    status = os.system(f'''
        export HCCL_DETERMINISTIC=true \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_matmul_reduce_scatter_normal \
               pytest -vra parallel/matmul_reduce_scatter.py::test_matmul_reduce_scatter_normal
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_matmul_reduce_scatter_normal_disable_kernel_backoff() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation disabling kernel backoff.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
                 mindspore.ops.MatMul and mindspore.ops.ReduceScatter forword calculation.
    """
    status = os.system(f'''
        export MS_DISABLE_KERNEL_BACKOFF=1 \
        && export HCCL_DETERMINISTIC=true \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_matmul_reduce_scatter_normal_disable_kernel_backoff \
               pytest -vra parallel/matmul_reduce_scatter.py::test_matmul_reduce_scatter_normal
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_matmul_reduce_scatter_dynamic() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of forword
                 calculation with inputs in static shapes.
    """
    status = os.system(f'''
        export HCCL_DETERMINISTIC=true \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_matmul_reduce_scatter_dynamic \
               pytest -vra parallel/matmul_reduce_scatter.py::test_matmul_reduce_scatter_dynamic
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_matmul_reduce_scatter_binary_cases() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
                 torch_npu.npu_mm_reduce_scatter_base forword calculation.
    """
    status = os.system(f'''
        export HCCL_DETERMINISTIC=true \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_matmul_reduce_scatter_binary_cases \
               pytest -vra parallel/matmul_reduce_scatter.py::test_matmul_reduce_scatter_binary_cases
    ''')
    assert status == 0


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='allcards',
    essential_mark='essential',
)
def test_matmul_reduce_scatter_binary_cases_disable_kernel_backoff() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation disabling kernel backoff.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
                 torch_npu.npu_mm_reduce_scatter_base forword calculation.
    """
    status = os.system(f'''
        export MS_DISABLE_KERNEL_BACKOFF=1 \
        && export HCCL_DETERMINISTIC=true \
        && msrun \
               --worker_num {WORKER_NUM} \
               --local_worker_num {WORKER_NUM} \
               --join True \
               --master_port 6221 \
               --log_dir test_matmul_reduce_scatter_binary_cases_disable_kernel_backoff \
               pytest -vra parallel/matmul_reduce_scatter.py::test_matmul_reduce_scatter_binary_cases
    ''')
    assert status == 0
