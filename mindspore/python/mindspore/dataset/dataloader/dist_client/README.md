
Client可以部署在任何一个节点，一般在训练进程中：
每一次 batch 的开始，都会
1. 与cordinator建立连接，提交注册信息；发送Client基础信息至coordinator，获取cluster server node集群信息；
2. 流式数据索引（通常是一个 batch 的 list）发送至coordinator，得到 索引 与 data node对应关系；
3. 每一次 RPC 调用（通常是 microbatch， 这里可能取 microbatch 的整数倍的处理完的样本回来） 从 server node 获取数据，暂存于数据缓存中；
4. 迭代器循环时，从数据缓存中拿一条数据。

在 `DistributedDataLoader` 中，这个流程被自动化：
- 每次 `__iter__` 开始时只向 coordinator 注册一次 client；
- 遍历到某个 microbatch 时，先检查本地缓存是否包含这些 index；
- 若存在缺失，就对缺失的 index 发起一次 RPC fetch；
- server node 可以返回超过当前 microbatch 的样本，loader 会把多余样本持续保存在缓存中，供后续 microbatch 命中使用。





> 数据缓存有点类似于滑动窗口的性质，因为一次可能取到 microbatch 的整数倍，但是当前只用到 microbatch 个样本，所以剩下的样本将会被缓存到缓存里面；
- 缓存可以简单用 python 队列什么的实现一下就行
- 分布式 dataloader 应该会有一个接口 get，要么从本地缓存拿到要么从远端 RPC 调用拿到

> 和 coordinator/servernode 的通信应该就用 RPC 而不是 ray.remote 这种形式
1. coordinator RPC 提供2个接口：1. 注册 client 的接口 2. 分配 servernode，这个需要一个 batch index 的list传过去 servernode
2. servernode RPC 提供一个接口：每次 microbatch 会传过去一个 index list，让 servernode 指定返回这些index处理完的样本，如果没有做完的话就同步等待 servernode 返回，servernode 要么就立即做要么就从他本地缓存的得到；servernode 一次返回可能不仅仅是这些 microbatch 而可能是还有后续的

设计到 ray.remote 的部分需要改写成为 RPC

> example

client 需要处理 [1,2,3,4]，分为两个 batch [1,2] 和 [3,4], batch size = 2，microbatch size = 1
在一次迭代内：

1. 处理 batch [1,2]：注册一次 coordinator，然后 RPC 分过去 index list，然后分配一个 Servernode 给 client
    - 处理 microbatch 1 [1]，往 servernode 发一个 RPC，可能获得 [1,2] 这两个处理完的样本回来
    - 处理 microbatch 2 [2]，本地缓存命中
2. 处理 batch [3,4]：注册一次 coordinator，然后 RPC 分过去 index list，然后分配一个 Servernode 给 client
    - 处理 microbatch 1 [3]，往 servernode 发一个 RPC，可能获得 [3] 这两个处理完的样本回来
    - 处理 microbatch 2 [4]，往 servernode 发一个 RPC，获得 [4]




> TODO PLAN

[x] 实现一个 dataloader 的 wrapper，可以进行分布式的传 index --> 得到处理完的数据
[] Client--servernode 通信走 RPC 路径
[] client 的缓存系统
[] client 的 batch-microbatch 逻辑细化
[] 采样算法

> FUTURE: 还需要考虑的问题：

DP/TP/PP/EP/SP 支持
多模态解耦时候的情况