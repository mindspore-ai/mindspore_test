mindspore.ops.TensorDump
========================

.. py:class:: mindspore.ops.TensorDump(input_output='out')

    将Tensor保存为numpy格式的npy文件。

    .. warning::
        参数input_output将不再支持参数值为'all'。

    .. note::
        在Ascend平台上的Graph模式下，可以通过设置环境变量 `MS_DUMP_SLICE_SIZE` 和 `MS_DUMP_WAIT_TIME` 解决在输出大Tesnor或输出Tensor比较密集的场景下算子执行失败的问题。

    参数：
        - **input_output** (str，可选) - 控制Tensordump行为模式的参数，可选的值为 ['out', 'in'] 中的一个，默认值： ``out`` 。

          对于算子A --> 重排算子 --> 算子B这样的情况，由于重排算子的插入，导致算子A的输出不再等价于算子B的输入。

          假设一种情况是算子A的计算结果既作为算子B的输入，也作为Tensordump算子的输入。则在该情况下，
          通过设置input_output参数可以实现不同的保存数据的需求：

          - 如果input_output参数设置为'out'，保存的数据仅包含算子A的输出分片。
          - 如果input_output参数设置为'in'，保存的数据将仅包含算子B的输入分片。

          当input_output参数被配置为'in'时，生成的输入分片所对应的npy文件命名格式为：fileName_dumpMode_dtype_id.npy。

          当input_output参数被配置为'out'时，生成的输出分片所对应的npy文件命名格式为：fileName_dtype_id.npy。

          - fileName：参数file的值 （若该参数传入时是一个使用者指定的路径，则fileName的值为路径的最后一级）。
          - dumpMode：input_output参数的值。
          - dtype：原始的数据类型。bfloat16类型数据保存在.npy文件中会被转换成float32类型。
          - id：一个自增的ID。

    输入：
        - **file** (str) - 要保存的文件路径。
        - **input_x** (Tensor) - 任意维度的Tensor。

    异常：
        - **TypeError** - 如果 `file` 不是str。
        - **TypeError** - 如果 `input_x` 不是Tensor。
