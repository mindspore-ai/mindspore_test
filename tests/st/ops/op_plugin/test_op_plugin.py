# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Mock tests for op plugin kernels."""

import os
import platform
import subprocess
from pathlib import Path
import pytest
import numpy as np

from tests.st.utils import test_utils

import mindspore as ms
from mindspore import Tensor
from mindspore import mint
from mindspore.ops.auto_generate.gen_ops_prim import expand_dims_view_op


def _configure_and_build_mock_plugin() -> str:
    """Configure and build the mock op plugin and return the built library path."""
    this_dir = Path(__file__).resolve().parent
    plugin_src_dir = this_dir / "mock_op_plugin"
    build_dir = plugin_src_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    repo_root = ms.__path__[0]
    # include path for custom_kernel_input_info.h
    include_dir = os.path.join(repo_root, "include", "mindspore", "ops", "kernel", "cpu", "custom", "kernel_mod_impl")

    system = platform.system().lower()

    cmake_args = [
        "cmake",
        "-S",
        str(plugin_src_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if system == "windows":
        include_flags = f"/I{include_dir}"
    else:
        include_flags = f"-I{include_dir}"
    cmake_args.append(f"-DCMAKE_CXX_FLAGS={include_flags}")

    # Configure
    subprocess.run(cmake_args, check=True)

    # Build
    build_cmd = ["cmake", "--build", str(build_dir)]
    if system == "windows":
        build_cmd += ["--config", "Release"]
    subprocess.run(build_cmd, check=True)

    # Locate built library
    exts = {
        "linux": ".so",
        "darwin": ".dylib",
        "windows": ".dll",
    }
    target_name = "mindspore_op_plugin_mock"
    target_ext = exts.get(system, ".so")

    candidates = []
    for p in build_dir.rglob(f"*{target_name}*{target_ext}"):
        # Prefer non-import libraries on Windows (exclude .lib/.exp)
        if p.suffix.lower() == target_ext:
            candidates.append(p)
    if not candidates:
        raise RuntimeError("Failed to locate built mock op plugin library")

    # Heuristic: pick the shortest path (usually the actual artifact, not intermediates)
    lib_path = str(sorted(candidates, key=lambda x: len(str(x)))[0])
    return lib_path

os.environ["MS_OP_PLUGIN_PATH"] = _configure_and_build_mock_plugin()

def set_mode(mode):
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

@test_utils.run_with_cell
def logical_and_forward_func(x, y):
    return mint.logical_and(x, y)

@test_utils.run_with_cell
def cumsum_ext_forward_func(x, dim):
    return mint.cumsum(x, dim)

@test_utils.run_with_cell
def inplace_relu_forward_func(x):
    mint.nn.functional.relu_(x)

@test_utils.run_with_cell
def view_func(x):
    out = expand_dims_view_op(x, 1)
    mint.nn.functional.relu_(out)
    return out


def test_cumsum(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([1, 2, 3, 4], ms.int64)
    dim = 0
    expect = np.cumsum(x.asnumpy(), dim)
    output = cumsum_ext_forward_func(x, dim)
    assert np.allclose(output.asnumpy(), expect)


def test_logical_and(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel when normal cpu kernelmod exists
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([True, False, True], ms.bool_)
    y = Tensor([True, True, False], ms.bool_)
    # there is a normal logical_and cpu kernelmod,
    # so mock logical_and op is implemented as logical_or
    # to ensure op plugin is used
    expect = np.logical_or(x.asnumpy(), y.asnumpy())
    output = logical_and_forward_func(x, y)
    assert np.allclose(output.asnumpy(), expect)


def test_inplace_relu(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for inplace op
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([-1.0, 4.0, -8.0, 2.0, -5.0, 9.0], ms.float32)
    expect = np.maximum(x.asnumpy(), 0.0)
    inplace_relu_forward_func(x)
    assert np.allclose(x.asnumpy(), expect)


def test_view_feature(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for view feature. Disabled for now
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([-1.0, 4.0, -8.0, 2.0, -5.0, 9.0], ms.float32)
    expected_x_after_inplace_relu = np.maximum(x.asnumpy(), 0.0)
    expect_view = expected_x_after_inplace_relu.reshape(6, 1)
    view = view_func(x)
    # TODO: fix the issue of view feature in op plugin
    # assert np.allclose(x.asnumpy(), expected_x_after_inplace_relu)
    # assert np.allclose(view.asnumpy(), expect_view)
    assert expect_view.shape == view.shape
