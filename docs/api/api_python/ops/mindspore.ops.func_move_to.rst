mindspore.ops.move_to
======================

.. py:function:: mindspore.ops.move_to(input, to="CPU", blocking="True")

    拷贝张量到目标设备，包含同步和异步两种方式，默认是同步方式，该接口当前仅支持图模式，并且jit_level为O0或O1。

    参数：
        - **input** (Union[Tensor, list[int], tuple[int]]) - 需要移动的输入Tensor，当输入为list和tuple时会转换为Tensor再进行拷贝。
        - **to** (str，可选) - 指定目标设备的名称，可选值为 ``"Ascend"``， ``"CPU"`` 。默认值： ``"CPU"`` 。
        - **blocking** (bool，可选) - 指定使用同步或异步拷贝，可选值为 ``"True"``， ``"False"`` 。默认值： ``"True"``，代表同步拷贝。

    返回：
        新的Tensor，存储在 `to` 指定的目标设备上，其类型和形状与输入 'input' 相同。

    异常：
        - **ValueError** - 如果 `to` 的值不为 ``"Ascend"`` 或 ``"CPU"`` 。
        - **ValueError** - 如果 `blocking` 的数据类型不为bool。
        - **ValueError** - 如果执行模式不是图模式或jit_level不为O0或O1。
