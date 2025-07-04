mindspore.common.initializer
=============================

初始化神经元参数。

.. py:class:: mindspore.common.initializer.Initializer(**kwargs)

    初始化器的抽象基类。

    .. note::
        Initializer本意是用于在并行模式中延迟Tensor的数据的初始化，而非初始化Tensor。如果必须使用Initializer来初始化Tensor，大多情况下需要在初始化之后使用 :func:`mindspore.Tensor.init_data` 。

    参数：
        - **kwargs** (dict) - `Initializer` 的关键字参数。

.. py:function:: mindspore.common.initializer.initializer(init, shape=None, dtype=mstype.float32)

    创建并初始化一个Tensor。

    参数：
        - **init** (Union[Tensor, str, Initializer, numbers.Number]) - 初始化方式。

          - **Tensor** - 用于初始化Tensor的Tensor。
          - **str** - `init` 是继承自 `Initializer` 的类的别名，实际使用时会调用相应的类。 `init` 的值可以是 ``"normal"`` 、 ``"ones"`` 或 ``"zeros"`` 等。
          - **Initializer** - `init` 是继承自 `Initializer` ，用于初始化Tensor的类。
          - **numbers.Number** - 用于初始化Tensor的常量。

        - **shape** (Union[tuple, list, int]) - 被初始化的Tensor的shape，默认值： ``None`` 。
        - **dtype** (mindspore.dtype) - 被初始化的Tensor的数据类型，默认值： ``mstype.float32`` 。

    返回：
        返回shape是入参 `shape` 的Tensor。如果 `shape` 是 ``None``，返回Tensor的shape是 `init` 的shape。

    异常：
        - **TypeError** - 参数 `init` 的类型不正确。
        - **ValueError** - 当 `init` 传入Tensor对象时， `init` 的shape与形参 `shape` 内的数值不一致。

.. py:class:: mindspore.common.initializer.TruncatedNormal(sigma=0.01, mean=0.0, a=-2.0, b=2.0)

    生成一个服从截断正态（高斯）分布的随机数组用于初始化Tensor。

    参数：
        - **sigma** (float) - 截断正态分布的标准差，默认值： ``0.01`` 。
        - **mean** (float) - 截断正态分布的平均值，默认值： ``0.0`` 。
        - **a** (float) - 截断区间的下界，默认值： ``-2.0`` 。
        - **b** (float) - 截断区间的上界，默认值： ``2.0`` 。

.. py:class:: mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)

    生成一个服从正态分布 :math:`{N}(\text{sigma}, \text{mean})` 的随机数组用于初始化Tensor。

    .. math::
        f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})

    参数：
        - **sigma** (float) - 正态分布的标准差，默认值： ``0.01`` 。
        - **mean** (float) - 正态分布的均值，默认值： ``0.0`` 。

.. py:class:: mindspore.common.initializer.Uniform(scale=0.07)

    生成一个服从均匀分布 :math:`{U}(-\text{scale}, \text{scale})` 的随机数组用于初始化Tensor。

    参数：
        - **scale** (float) - 均匀分布的边界，默认值： ``0.07`` 。

.. py:class:: mindspore.common.initializer.HeUniform(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu')

    生成一个服从HeKaiming均匀分布 :math:`{U}(-\text{boundary}, \text{boundary})` 的随机数组用于初始化Tensor，其中：

    .. math::
        boundary = \text{gain} \times \sqrt{\frac{3}{fan\_mode}}

    :math:`gain` 是一个可选的缩放因子。如果参数 `mode` 是 ``'fan_in'`` ，则公式中的 :math:`fan\_mode` 是权重Tensor中输入单元的数量。如果参数 `mode` 是 ``'fan_out'`` ，则公式中的 :math:`fan\_mode` 是权重Tensor中输出单元的数量。

    有关HeUniform算法，详情可参考 https://arxiv.org/abs/1502.01852。

    参数：
        - **negative_slope** (int, float, bool) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数 'leaky_relu'），默认值： ``0`` 。
        - **mode** (str) - 可选 ``'fan_in'`` 或 ``'fan_out'`` 。 ``'fan_in'`` 会保留前向传递中权重方差的量级， ``'fan_out'`` 会保留反向传递的量级。默认值： ``'fan_in'`` 。
        - **nonlinearity** (str) - 非线性激活函数，推荐使用 ``'relu'`` 或 ``'leaky_relu'`` ，默认值： ``'leaky_relu'`` 。

.. py:class:: mindspore.common.initializer.HeNormal(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu')

    生成一个服从HeKaiming正态分布 :math:`{N}(0, \text{sigma}^2)` 的随机数组用于初始化Tensor，其中：

    .. math::
        sigma = \frac{gain} {\sqrt{fan\_mode}}

    :math:`gain` 是一个可选的缩放因子。如果参数 `mode` 是 ``'fan_in'`` ，则公式中的 :math:`fan\_mode` 是权重Tensor中输入单元的数量。如果参数 `mode` 是 ``'fan_out'`` ，则公式中的 :math:`fan\_mode` 是权重Tensor中输出单元的数量。

    HeNormal 算法的详细信息，请查看 https://arxiv.org/abs/1502.01852。

    参数：
        - **negative_slope** (int, float) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数 'leaky_relu'），默认值： ``0`` 。
        - **mode** (str) - 可选 ``'fan_in'`` 或 ``'fan_out'`` 。 ``'fan_in'`` 会保留前向传递中权重方差的量级， ``'fan_out'`` 会保留反向传递的量级。默认值： ``'fan_in'`` 。
        - **nonlinearity** (str) - 非线性激活函数，推荐使用 ``'relu'`` 或 ``'leaky_relu'`` ，默认值： ``'leaky_relu'`` 。

.. py:class:: mindspore.common.initializer.XavierNormal(gain=1)

    生成一个服从Xavier正态分布的随机数组 :math:`{N}(0, \text{sigma}^2)` 用于初始化Tensor，其中：

    .. math::
        sigma = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    :math:`gain` 是一个可选的缩放因子。:math:`n_{in}` 为权重Tensor中输入单元的数量，:math:`n_{out}` 为权重Tensor中输出单元的数量。

    有关 XavierNormal 算法的详细信息，请查看 http://proceedings.mlr.press/v9/glorot10a.html。

    参数：
        - **gain** (float) - 可选的缩放因子，默认值： ``1`` 。

.. py:class:: mindspore.common.initializer.XavierUniform(gain=1)

    生成一个服从Xavier均匀分布 :math:`{U}(-\text{boundary}, \text{boundary})` 的随机数组用于初始化Tensor，均匀分布的取值范围为[-boundary, boundary]，其中：

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    :math:`gain` 是一个可选的缩放因子。:math:`n_{in}` 为权重Tensor中输入单元的数量，:math:`n_{out}` 为权重Tensor中输出单元的数量。

    有关 XavierUniform 算法的详细信息，请查看 http://proceedings.mlr.press/v9/glorot10a.html。

    参数：
        - **gain** (float) - 可选的缩放因子，默认值： ``1`` 。

.. py:class:: mindspore.common.initializer.One(**kwargs)

    生成一个值全为1的常量数组用于初始化Tensor。

.. py:class:: mindspore.common.initializer.Zero(**kwargs)

    生成一个值全为0的常量数组用于初始化Tensor。

.. py:class:: mindspore.common.initializer.Constant(value)

    生成一个常量数组用于初始化Tensor。

    参数：
        - **value** (Union[int, numpy.ndarray]) - 用于初始化的常数值或者数组。

.. automodule:: mindspore.common.initializer
    :members:

.. py:class:: mindspore.common.initializer.Identity(**kwargs)

    生成一个二维的单位矩阵用于初始化Tensor。

    异常：
        - **ValueError** - 被初始化的Tensor的维度不等于2。

.. py:class:: mindspore.common.initializer.Sparse(sparsity, sigma=0.01)

    生成一个二维的稀疏矩阵用于初始化Tensor。矩阵非0的位置的值服从正态分布 :math:`{N}(0, sigma)` 。

    参数：
        - **sparsity** (float) - 矩阵每列中元素被置0的比例。
        - **sigma** (float) - 正态分布的标准差，默认值： ``0.01`` 。

    异常：
        - **ValueError** - 被初始化的Tensor的维度不等于2。

.. py:class:: mindspore.common.initializer.Dirac(groups=1)

    利用Dirac delta函数生成一个矩阵用于初始化Tensor。Dirac初始化方式通常用于卷积层，会尽可能多的保留输入的特性。

    参数：
        - **groups** (int) - 卷积层中的分组数量，每个组执行相同的初始化。默认值： ``1`` 。

    异常：
        - **ValueError** - 被初始化的Tensor的维度不在[3, 4, 5]的范围内。
        - **ValueError** - 被初始化的Tensor的第一个维度不能被groups整除。

.. py:class:: mindspore.common.initializer.Orthogonal(gain=1.)

    生成一个正交或半正交矩阵用于初始化Tensor。被初始化的Tensor的维度至少为2。如果维度大于2，多余的维度将会被展平。

    参数：
        - **gain** (float) - 可选的比例因子，默认值为 ``1.0`` 。

    异常：
        - **ValueError** - 被初始化的Tensor的维度小于2。

.. py:class:: mindspore.common.initializer.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')

    生成一个随机的矩阵用于初始化Tensor。

    当 `distribution` 是 ``'truncated_normal'`` 或者 ``'untruncated_normal'`` 时，矩阵中的值将服从均值为0，标准差\
    为 :math:`stddev = \sqrt{\frac{scale}{n}}` 的截断或者非截断正态分布。

    如果 `mode` 是 ``'fan_in'``， :math:`n` 是输入单元的数量；如果 `mode` 是 ``'fan_out'``， :math:`n` 是输出单元的数量；如果 `mode` 是 ``'fan_avg'`` ， :math:`n` 是输入输出单元数量的均值。

    当 `distribution` 是 ``'uniform'`` 时，矩阵中的值将服从均匀分布 :math:`[-\sqrt{\frac{3*scale}{n}}, \sqrt{\frac{3*scale}{n}}]`。

    参数：
        - **scale** (float) - 比例因子，默认值； ``1.0`` 。
        - **mode** (str) - 其值应为 ``'fan_in'`` 、 ``'fan_out'`` 或者 ``'fan_avg'`` ，默认值： ``'fan_in'`` 。
        - **distribution** (str) - 用于采样的分布类型。其值应为 ``'uniform'`` 、 ``'truncated_normal'`` 或 ``'untruncated_normal'`` ，默认值： ``'truncated_normal'`` 。

    异常：
        - **ValueError** - `scale` 小于等于0。
        - **ValueError** - `mode` 不是 ``'fan_in'`` 、 ``'fan_out'`` 或者 ``'fan_avg'`` 。
        - **ValueError** - `distribution` 不是 ``'truncated_normal'`` 、 ``'untruncated_normal'`` 或者 ``'uniform'`` 。



