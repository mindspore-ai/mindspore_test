mindspore.dataset.text.SlidingWindow
====================================

.. py:class:: mindspore.dataset.text.SlidingWindow(width, axis=0)

    从给定数据构建Tensor（当前仅支持1维），其中 `axis` 轴上的每个元素都是从对应位置开始的指定宽度的数据切片。

    参数：
        - **width** (int) - 窗口的宽度。其值必须大于零。
        - **axis** (int, 可选) - 沿着哪一个轴计算滑动窗口。默认值： ``0`` 。

    异常：
        - **TypeError** - 参数 `width` 的类型不为int。
        - **ValueError** - 参数 `width` 的值不为正数。
        - **TypeError** - 参数 `axis` 的类型不为int。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
