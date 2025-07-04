mindspore.train.Recall
=======================

.. py:class:: mindspore.train.Recall(eval_type='classification')

    计算数据分类的召回率，包括单标签场景和多标签场景。

    Recall类创建两个局部变量 :math:`\text{true_positive}` 和 :math:`\text{false_negative}` 用于计算召回率。计算方式为：

    .. math::
        \text{recall} = \frac{\text{true_positive}}{\text{true_positive} + \text{false_negative}}

    .. note::
        在多标签情况下， :math:`y` 和 :math:`y_{pred}` 的元素必须为0或1。

    参数：
        - **eval_type** (str) - 支持 ``'classification'`` （单标签分类） 和 ``'multilabel'`` （多标签分类）。默认值： ``'classification'`` 。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval(average=False)

        计算召回率。

        参数：
            - **average** (bool) - 指定是否计算平均召回率。默认值： ``False`` 。

        返回：
            numpy.float64，计算结果。

    .. py:method:: update(*inputs)

        使用预测值 `y_pred` 和真实标签 `y` 更新局部变量。

        参数：
            - **inputs** - 输入 `y_pred` 和 `y`。 `y_pred` 和 `y` 支持Tensor、list或numpy.ndarray类型。

              - 对于 ``'classification'`` 场景， `y_pred` 在大多数情况下由范围 :math:`[0, 1]` 中的浮点数组成，shape为 :math:`(N, C)` ，其中 :math:`N` 是样本数， :math:`C` 是类别数。 `y` 由整数值组成，如果是one_hot编码格式，shape是 :math:`(N, C)` ；如果是类别索引，shape是 :math:`(N,)` 。
              - 对于 ``'multilabel'`` 场景， `y_pred` 和 `y` 只能是值为0或1的one-hot编码格式，其中值为1的索引表示正类别。 `y_pred` 和 `y` 的shape都是 :math:`(N, C)` 。

        异常：
            - **ValueError** - inputs数量不是2。
