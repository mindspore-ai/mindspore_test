mindspore.ops.TensorDump
========================

.. py:class:: mindspore.ops.TensorDump(input_output='out')

    将Tensor保存为numpy格式的npy文件。

    .. warning::
        - 如果在短时间内保存大量数据，可能会导致设备端内存溢出。可以考虑对数据进行切片，以减小数据规模。
        - 由于数据保存是异步处理的，当数据量过大或主进程退出过快时，可能出现数据丢失的问题，需要主动控制主进程销毁时间，例如使用sleep。

    参数：
        - **input_output** (str，可选) - 控制Tensordump行为模式的参数，可选的值为 ['out', 'in', 'all'] 中的一个，默认值： ``out`` 。

          对于算子A --> 重排算子 --> 算子B这样的情况，由于重排算子的插入，导致算子A的输出不再等价于算子B的输入。

          假设一种情况是算子A的计算结果既作为算子B的输入，也作为Tensordump算子的输入。则在该情况下，
          通过设置input_output参数可以实现不同的保存数据的需求：

          - 如果input_output参数设置为'out'，保存的数据仅包含算子A的输出分片。
          - 如果input_output参数设置为'all'，保存的数据将包含算子A的输出分片以及算子B的输入分片。
          - 如果input_output参数设置为'in'，保存的数据将仅包含算子B的输入分片。

          当input_output参数被配置为'all'或'in'时，生成的输入分片所对应的npy文件命名格式为：id_fileName_cNodeID_dumpMode_rankID.npy。

          当input_output参数被配置为'all'或'out'时，生成的输出分片所对应的npy文件命名格式为：id_fileName.npy。

          - id：一个自增的ID。
          - fileName：参数file的值 （若该参数传入时是一个使用者指定的路径，则fileName的值为路径的最后一级）。
          - cNodeID：该Tensordump节点在step_parallel_end.ir文件中的节点编号。
          - dumpMode：input_output参数的值。
          - rankID：逻辑卡号。

    输入：
        - **file** (str) - 要保存的文件路径。
        - **input_x** (Tensor) - 任意维度的Tensor。

    异常：
        - **TypeError** - 如果 `file` 不是str。
        - **TypeError** - 如果 `input_x` 不是Tensor。
