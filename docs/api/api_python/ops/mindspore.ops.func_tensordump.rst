mindspore.ops.tensordump
========================

.. py:function:: mindspore.ops.tensordump(file_name, tensor, mode)

    将Tensor保存为Numpy的npy格式的文件。

    在并行的场景下，该算子会保存不同计算卡上的数据分片。

    在昇腾平台的静态图并行模式下，代码 算子A --> 算子B 可能会被编译为 算子A --> 重排算子 --> 算子B。

    注：重排算子指在静态图并行场景中，由于设备间通信，算子切分策略而引入的算子。

    对于 算子A --> 算子B 的情况，算子A计算结果的输出等于算子B的输入。

    然而对于算子A --> 重排算子 --> 算子B这样的情况，由于重排算子的插入，导致算子A的输出不再等价于算子B的输入。

    假设一种情况是算子A的计算结果既作为算子B的输入，也作为tensordump算子的输入。则在该情况下，
    通过设置mode参数可以实现不同的保存数据的需求：

    - 如果mode参数设置为'out'，保存的数据仅包含算子A的输出分片。
    - 如果mode参数设置为'all'，保存的数据将包含算子A的输出分片以及算子B的输入分片。
    - 如果mode参数设置为'in'，保存的数据将仅包含算子B的输入分片。

    当mode参数被配置为'all'或'in'时，生成的输入分片所对应的npy文件命名格式为：id_fileName_cNodeID_dumpMode_rankID.npy。

    当mode参数被配置为'all'或'out'时，生成的输出分片所对应的npy文件命名格式为：id_filename.npy。

    - id：一个自增的ID。
    - fileName：参数file_name的值 （若该参数传入时是一个使用者指定的路径，则fileName的值为路径的最后一级）。
    - cNodeID：该tensordump节点在step_parallel_end.ir文件中的节点编号
    - dumpMode：mode参数的值。
    - rankID：逻辑卡号。

    .. note::
        - 当前该算子不支持在控制流中使用。
        - 如果当前的并行模式为STAND_ALONE，参数mode只能设置为'out'。
        - 如使用该算子时不设置参数mode，其默认值为'out'。
    
    参数：
        - **file_name** (str) - npy文件的保存路径。
        - **tensor** (Tensor) - 输入的张量。
        - **mode** (str) - 控制tensordump行为模式的参数，可选的值为 ['out', 'in', 'all'] 中的一个。

    异常：
        - **TypeError** - `file_name` 不是一个str类型。
        - **TypeError** - `tensor` 不是一个Tensor类型。
        - **TypeError** - `mode` 不是一个str类型。
        - **ValueError** - `mode` 的值不是 ['out', 'in', 'all'] 之中的一个。

    样例：

    .. note:: 
        使用msrun命令运行下面的例子：msrun --worker_num=2 --local_worker_num=2 --master_port=11450
        --log_dir=msrun_log --join=True --cluster_time_out=300 tensordump_example.py

