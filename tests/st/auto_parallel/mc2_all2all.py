import numpy as np
import os
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore as ms
import mindspore.communication.management as D
from mindspore.common.initializer import initializer, Tensor
from mindspore import context, Parameter
import mindspore.ops as ops

os.environ["HCCL_OP_EXPANSION_MODE"] = "HOST"


class Linear(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.transpose_b = transpose_b
        self.expert_flag = True
        self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                name="weight")
        self.matmul = ops.BatchMatMul(transpose_b=transpose_b)

        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [1, self.expert_num, 1, out_channels], param_init_type),
                                  name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = ops.Add()

        self.dtype = compute_dtype
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag:
            x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = ops.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = ops.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, out_strategy_matmul=None):
        self.matmul.shard(in_strategy=strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class MoEFFNet(nn.Cell):
    def __init__(self, hidden_size, ffn_hidden_size, expert_num, dp, ep, mp,
                 sp=False, has_bias=False, transpose_b=False,
                 alltoallallgatherbatchmatmul_withoutsilu=False,
                 alltoallallgatherbatchmatmul_withsilu=False,
                 batchmatmulreducescatteralltoall=False):
        super(MoEFFNet, self).__init__()
        input_size = hidden_size
        output_size = ffn_hidden_size
        param_init_type = mstype.float16
        compute_dtype = mstype.float16
        self.dp = dp
        self.use_sp = sp
        outer_bs = dp
        if sp:
            outer_bs = dp * ep
        if (not alltoallallgatherbatchmatmul_withoutsilu and
                not alltoallallgatherbatchmatmul_withsilu and
                not batchmatmulreducescatteralltoall):
            raise ValueError("Have not chose any mc2_fusion pattern !!!")
        self.alltoallallgatherbatchmatmul_withoutsilu = \
            alltoallallgatherbatchmatmul_withoutsilu
        self.alltoallallgatherbatchmatmul_withsilu = \
            alltoallallgatherbatchmatmul_withsilu
        self.batchmatmulreducescatteralltoall = \
            batchmatmulreducescatteralltoall
        self.mapping = Linear(in_channels=input_size,
                              out_channels=output_size,
                              has_bias=has_bias,
                              transpose_b=transpose_b,
                              expert_num=expert_num,
                              outer_batch=outer_bs,
                              param_init_type=param_init_type,
                              compute_dtype=compute_dtype)

        self.projection = Linear(in_channels=output_size,
                                 out_channels=input_size,
                                 has_bias=has_bias,
                                 transpose_b=transpose_b,
                                 expert_num=expert_num,
                                 outer_batch=outer_bs,
                                 param_init_type=param_init_type,
                                 compute_dtype=compute_dtype)

        if transpose_b:
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                               strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)))
            self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)),
                                  strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)))
        else:
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                               strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)))
            self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                  strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)))
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.stride_slice_ep = ops.StridedSlice().shard(((ep, 1, 1, 1),))
        self.stride_slice_ep_mp = ops.StridedSlice().shard(((ep, 1, mp, 1),))
        self.stride_slice_dp_ep = ops.StridedSlice().shard(((dp, ep, 1, 1, 1),))
        self.stride_slice_dp_ep_mp = ops.StridedSlice().shard(((dp, ep, 1, mp, 1),))
        self.stride_slice_sp_ep = ops.StridedSlice().shard(((dp, 1, ep, 1, 1),))
        self.stride_slice_sp_ep_mp = ops.StridedSlice().shard(((dp, 1, ep, mp, 1),))

        self.stride_slice_dp_mp = ops.StridedSlice().shard(((dp, 1, ep, mp, 1),))
        self.stride_slice_ep_mp = ops.StridedSlice().shard(((dp, ep, 1, mp, 1),))

    def construct(self, x):
        if self.alltoallallgatherbatchmatmul_withoutsilu:
            output = self.construct_alltoallallgatherbatchmatmul_withoutsilu(x)
        elif self.alltoallallgatherbatchmatmul_withsilu:
            output = self.construct_alltoallallgatherbatchmatmul_withsilu(x)
        elif self.batchmatmulreducescatteralltoall:
            output = self.construct_batchmatmulreducescatteralltoall(x)
        return output

    def construct_alltoallallgatherbatchmatmul_withoutsilu(self, x):
        x_shape = self.shape(x)
        x = self.stride_slice_dp_mp(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        x = self.stride_slice_ep_mp(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        x = self.stride_slice_dp_ep(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        hidden = self.mapping(x)
        return hidden

    def construct_alltoallallgatherbatchmatmul_withsilu(self, x):
        x_shape = self.shape(x)
        x = self.stride_slice_dp_mp(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        x = self.stride_slice_ep_mp(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        x = self.stride_slice_dp_ep(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        hidden = self.mapping(x)
        hidden = ops.silu(hidden)
        return hidden

    def construct_batchmatmulreducescatteralltoall(self, x):
        x_shape = (1, 16, 2, 512, 4096)
        output = self.projection(x)
        output1 = self.reshape(output, x_shape)
        output2 = self.stride_slice_dp_ep_mp(output1, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        output2 = self.stride_slice_dp_mp(output2, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        return output2



def test_mc2_alltoall_allgather_batchmatmul_withoutsilu():
    ms.set_seed(seed=100)
    os.environ["HCCL_OP_EXPANSION_MODE"] = "HOST"
    os.environ["HCCL_IF_BASE_PORT"] = "30000"

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"}) # KBK
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel",
        dataset_strategy="full_batch",
        enable_alltoall=True
    )
    D.init()

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 2
    mp = 2

    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)

    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp,
                   alltoallallgatherbatchmatmul_withoutsilu=True)
    expect_out = net(x).asnumpy()

    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})

    mc2_net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp,
                       alltoallallgatherbatchmatmul_withoutsilu=True)
    mc2_out = mc2_net(x).asnumpy()

    assert np.allclose(expect_out, mc2_out, 1e-3, 1e-3)

def test_mc2_alltoall_allgather_batchmatmul_withsilu():
    ms.set_seed(seed=100)
    os.environ["HCCL_OP_EXPANSION_MODE"] = "HOST"
    os.environ["HCCL_IF_BASE_PORT"] = "30016"

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"}) # KBK
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel",
        dataset_strategy="full_batch",
        enable_alltoall=True
    )
    D.init()

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 2
    mp = 2

    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)

    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp,
                   alltoallallgatherbatchmatmul_withsilu=True)
    expect_out = net(x).asnumpy()

    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})

    mc2_net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp,
                       alltoallallgatherbatchmatmul_withsilu=True)
    mc2_out = mc2_net(x).asnumpy()

    assert np.allclose(expect_out, mc2_out, 1e-3, 1e-3)
