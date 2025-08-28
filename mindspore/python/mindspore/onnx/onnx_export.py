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

"""Model export to ONNX."""
from __future__ import absolute_import
from __future__ import division

import os
import stat

import mindspore.nn as nn
from mindspore import log as logger
from mindspore._checkparam import check_input_dataset
from mindspore import _checkparam as Validator
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.train.serialization import _calculation_net_size
from mindspore.dataset.engine.datasets import Dataset

PROTO_LIMIT_SIZE = 1024 * 1024 * 2


def export(net, *inputs, file_name, input_names=None, output_names=None, export_params=True,
           keep_initializers_as_inputs=False, dynamic_axes=None):
    """
    Export the MindSpore network into an ONNX model.

    Note:
        - Support exporting network larger than 2GB. When the network exceeds 2GB,
          parameters are saved in additional binary files stored in the same directory as the ONNX file.
        - When `file_name` does not have a suffix, the system will automatically add the suffix `.onnx` .

    Args:
        net (Union[Celel, function]): MindSpore network.
        inputs (Union[Tensor, list, tuple, Number, bool]): It represents the inputs of the `net` , if the network has
            multiple inputs, set them together.
        file_name (str): File name of the model to be exported.
        input_names (list, optional): Names to assign to the input nodes of the graph, in order. Default: ``None`` .
        output_names (list, optional): Names to assign to the output nodes of the graph, in order. Default: ``None`` .
        export_params (bool, optional): If false, parameters (weights) will not be exported,
            parameters will add input nodes as input of the graph. Default: ``True`` .
        keep_initializers_as_inputs (bool, optional): If True, all the initializers (model parameters/weights) will
            add as inputs to the graph. This allows modifying any or all weights when running the exported ONNX model.
            Default: ``False`` .
        dynamic_axes (dict[str, dict[int, str]], optional): To specify axes of input tensors as dynamic (at runtime).
            Default: ``None`` .

            - Set a dict with scheme: {input_node_name: {axis_index:axis_name}},
              for example, {"input1": {0:"batch_size", 1: "seq_len"}, "input2": {{0:"batch_size"}}.
            - By default, the shapes of all input tensors in the exported model exactly match those specified in
              `inputs`.

    Raises:
        ValueError: If the parameter `net` is not :class:`mindspore.nn.Cell`.
        ValueError: If the parameter `input_names` is not list type.
        ValueError: If the parameter `output_names` is not list type
        ValueError: If the parameter `dynamic_axes` is not dict type.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.onnx.export(net, input_tensor, file_name='lenet.onnx', input_names=['input1'], output_names=['output1'])

    """
    Validator.check_file_name_by_regular(file_name)
    logger.info("exporting model file:%s format:%s.", file_name, "ONNX")
    if input_names is not None and not isinstance(input_names, list):
        raise ValueError(
            f"For 'onnx.export', the type of 'input_names' must be a list, but got '{type(input_names)}'")
    if output_names is not None and not isinstance(output_names, list):
        raise ValueError(
            f"For 'onnx.export', the type of 'output_names' must be a list, but got '{type(output_names)}'")
    if dynamic_axes is not None and not isinstance(dynamic_axes, dict):
        raise ValueError(
            f"For 'onnx.export', the type of 'dynamic_axes' must be a directory, but got '{type(dynamic_axes)}'")

    extra_save_params = False

    if check_input_dataset(*inputs, dataset_type=Dataset):
        raise ValueError(f"Can not support dataset as inputs to export ONNX model.")

    file_name = os.path.realpath(file_name)

    if not isinstance(net, nn.Cell):
        raise ValueError(f"Export ONNX format model only support nn.Cell object, but got {type(net)}.")

    cell_mode = net.training
    net.set_train(mode=False)
    total_size = _calculation_net_size(net)
    if total_size > PROTO_LIMIT_SIZE:
        logger.warning('Network size is: {}G, it exceeded the protobuf: {}G limit, now parameters in network are saved '
                       'in external data files.'.format(total_size / 1024 / 1024, PROTO_LIMIT_SIZE / 1024 / 1024))
        extra_save_params = True
    phase_name = 'export.onnx'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)

    if not file_name.endswith('.onnx'):
        file_name += ".onnx"
    abs_file_name = os.path.abspath(file_name)
    if os.path.exists(abs_file_name):
        os.chmod(abs_file_name, stat.S_IWUSR)
    else:
        dir_path = os.path.dirname(abs_file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, mode=0o700, exist_ok=True)
        os.chmod(dir_path, 0o700)

    abs_file_dir = ""
    if extra_save_params:
        abs_file_dir = os.path.dirname(abs_file_name)

    onnx_stream = _executor._get_onnx_func_graph_proto(obj=net, exec_id=graph_id, input_names=input_names,
                                                       output_names=output_names, export_params=export_params,
                                                       keep_initializers_as_inputs=keep_initializers_as_inputs,
                                                       dynamic_axes=dynamic_axes, extra_save_params=extra_save_params,
                                                       save_file_dir=abs_file_dir)
    with open(abs_file_name, 'wb') as f:
        f.write(onnx_stream)
        os.chmod(abs_file_name, stat.S_IRUSR)

    net.set_train(mode=cell_mode)
