mindspore.runtime.set_cpu_affinity
===================================

.. py:function:: mindspore.runtime.set_cpu_affinity(enable_affinity, affinity_cpu_list=None)

    使能线程级绑核功能，给MindSpore的主要模块（主线程、pynative、runtime、minddata）分配特定的CPU核，防止MindSpore线程抢占CPU导致的性能不稳定情况。

    .. note::
        - 提供两种绑核模式：

          1. 依据环境上可用的CPU、NUMA节点和设备资源，自动化生成绑核策略，进行线程级绑核。
          2. 依据 `affinity_cpu_list` 传入的自定义绑核策略，进行线程级绑核。
        - 自动化生成绑核策略场景会调用系统命令去获取环境上的CPU、NUMA节点和设备资源，由于环境差异，一些命令无法成功执行；按照环境上可获取资源，生成的自动化绑核策略会有所差异：

          1. `cat /sys/fs/cgroup/cpuset/cpuset.cpus`，获取环境上可用的CPU资源；若执行该命令失败，绑核功能不会生效。
          2. `npu-smi info -m`，获取环境上可用的NPU资源；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          3. `npu-smi info -t board -i {NPU_ID} -c {CHIP_ID}`，根据设备逻辑ID去获取NPU的详细信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          4. `lspci -s {PCIe_No} -vvv`，获取环境上设备的硬件信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          5. `lscpu`，获取环境上CPU与NUMA节点的信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。

    参数：
        - **enable_affinity** (bool) - 开关线程级绑核功能。
        - **affinity_cpu_list** (dict，可选) - 指定自定义的绑核策略。传入字典的key需要为字符串 ``"deviceX"`` 格式，value需要为列表 ``["cpuidX-cpuidY"]`` 格式。默认值： ``None``，即使用依据环境自动化生成的绑核策略。允许传入空字典 ``{}``，这种情况下会依据环境自动化生成的绑核策略。

    异常：
        - **TypeError** - 参数 `enable_affinity` 不是一个 ``bool``。
        - **TypeError** - 参数 `affinity_cpu_list` 既不是一个字典，也不是一个 ``None``。
        - **ValueError** - 参数 `affinity_cpu_list` 的key不是一个字符串。
        - **ValueError** - 参数 `affinity_cpu_list` 的key不符合 ``"deviceX"`` 格式。
        - **ValueError** - 参数 `affinity_cpu_list` 的value不是一个列表。
        - **ValueError** - 参数 `affinity_cpu_list` 的value中的元素不是一个字符串。
        - **ValueError** - 参数 `affinity_cpu_list` 的value中的元素不符合 ``["cpuidX-cpuidY"]`` 格式。
        - **RuntimeError** - 自动化生成绑核策略或者自定义指定绑核策略场景，分配给每个设备的CPU核数量小于7个。
        - **RuntimeError** - 自定义指定绑核策略场景，分配给某个设备的CPU在环境中不可用。
        - **RuntimeError** - 重复调用了 `mindspore.runtime.set_cpu_affinity` 接口。
