import mindspore
from mindspore import mint
from mindspore import ops
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspore.nn.cell import Cell
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormReduceGrad
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormElemtGrad


class SyncBatchNormInner(Cell):
    def __init__(self, self_num_features, self_world_size):
        super(SyncBatchNormInner, self).__init__()
        self.num_features = self_num_features
        self.world_size = self_world_size
        self.batch_norm_reduce_grad = BatchNormReduceGrad()
        self.batch_norm_elemt_grad = BatchNormElemtGrad()

    def construct(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        assert self.world_size == world_size
        if not input.is_contiguous():
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        size = int(input.numel() // input.shape[1])
        if size == 1 and world_size < 2:
            raise ValueError(
                'Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate mean/invstd for input.
        mean, invstd = mint.batch_norm_stats(input, eps)
        count = mint.full((1,), input.numel() //
                          input.shape[1], dtype=mean.dtype)

        num_channels = input.shape[1]
        assert self.num_features == num_channels
        # C, C, 1 -> (2C + 1)
        combined = mint.cat([mean, invstd, count], dim=0)
        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # world_size * (2C + 1)
        combined, _ = all_gather_into_tensor(combined, process_group)
        combined = ops.reshape(combined, [world_size, -1])
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = mint.split(
            combined, num_channels, dim=1)
        # calculate global mean & invstd
        mean, invstd = mint.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        # apply element-wise normalization
        out = mint.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return (out, mean, invstd, count_all.view(-1))

    def bprop(self, input_x, weight, bias, running_mean, running_var, eps, momentum,
              process_group, world_size, output, doutput):
        _, mean_param, invstd_param, count_all_param = output
        dout, _, _, _ = doutput

        # 图模式不支持is_contiguous
        if not dout.is_contiguous():
            dout = dout.contiguous()

        grad_input = grad_weight = grad_bias = None

        # 框架上不支持动态获取tensor/parameter的requires_grad
        # 临时写死，对所有输入都进行求导
        inputG = True
        weightG = True
        biasG = True

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = self.batch_norm_reduce_grad(
            dout,
            input_x,
            mean_param,
            invstd_param,
            weight,
            inputG,
            weightG,
            biasG
        )

        if inputG:
            # synchronizing stats used to calculate input gradient.
            num_channels = sum_dy.shape[0]
            combined = mint.cat([sum_dy, sum_dy_xmu], dim=0)

            new_combined, _ = mindspore.communication.comm_func.all_reduce(
                combined, group=process_group)

            sum_dy, sum_dy_xmu = mint.split(new_combined, num_channels)

            # backward pass for gradient calculation
            grad_input = self.batch_norm_elemt_grad(
                dout,
                input_x,
                mean_param,
                invstd_param,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_all_param
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not weightG:
            grad_weight = None

        if weight is None or not biasG:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _SyncBatchNorm(Cell):
    def __init__(self, num_features, world_size, dtype=mindspore.float32):
        super(_SyncBatchNorm, self).__init__()
        self.num_features = num_features
        self.world_size = world_size
        self.inner = SyncBatchNormInner(self.num_features, self.world_size)

    def construct(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        res = self.inner(input, weight, bias, running_mean,
                         running_var, eps, momentum, process_group, world_size)
        output, _, _, _ = res
        return output
