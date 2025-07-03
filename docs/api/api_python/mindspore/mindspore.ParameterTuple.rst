mindspore.ParameterTuple
========================

.. py:class:: mindspore.ParameterTuple

    继承于tuple，用于管理多个Parameter。

    .. note::
        该类把网络参数存储到参数元组集合中。

    .. py:method:: clone(prefix, init='same')

        逐个对ParameterTuple中的Parameter进行克隆，生成新的ParameterTuple。

        参数：
            - **prefix** (str) - Parameter的namespace，此前缀将会被添加到ParameterTuple中的Parameter的name属性中。
            - **init** (Union[Tensor, str, numbers.Number]) - 对ParameterTuple中Parameter的shape和类型进行克隆，并根据传入的 `init` 设置数值。默认值： ``'same'`` 。

              - 如果 `init` 是 `Tensor` ，则新参数的数值与该Tensor相同。
              - 如果 `init` 是 `str` ，则按照 `Initializer` 模块中对应的同名的初始化方法设置新参数的数值。若 `init` 的值为 ``'same'``，则新参数的数值与原Parameter相同。
              - 如果 `init` 是 `numbers.Number` ，则设置新参数的数值为该值。

        返回：
            新的参数元组。
