import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormReduceGrad
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormElemtGrad
from mindspore.communication import GlobalComm
from mindspore.ops import ReduceOp
from mindspore._c_expression import Tensor as Tensor_
from mindspore.communication._comm_helper import _get_size_helper, HCCL_WORLD_COMM_GROUP
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import operations as P
from mindspore import ops, mint


DEFAULT_WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP

batch_norm_reduce_grad = BatchNormReduceGrad()
batch_norm_elemt_grad = BatchNormElemtGrad()
shape = P.Shape()


def _deal_comm_outputs(output, async_op):
    if isinstance(output, tuple):
        if not async_op:
            output[1].wait()
            return output[0]
        return output

    if not async_op:
        return output
    return output


def get_group_size(group=GlobalComm.WORLD_COMM_GROUP):
    if not isinstance(group, str):
        raise TypeError("For 'get_group_size', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_size_helper(group=_get_group(group))


def _contiguous(tensor):
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        tensor = tensor.contiguous()
    return tensor


def _get_group(group):
    """Return the world communication group if the `group` is `DEFAULT_WORLD_COMM_GROUP`."""
    if group == DEFAULT_WORLD_COMM_GROUP:
        return GlobalComm.WORLD_COMM_GROUP
    return group


def all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP, async_op=False):
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError(
            "For all_gather_into_tensor, the input tensor must be tensor")
    group = _get_group(group)
    tensor = _contiguous(tensor)
    all_gather_op = _get_cache_prim(P.AllGather)(group=group)
    output = all_gather_op(tensor)
    return _deal_comm_outputs(output, async_op)


def all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP, async_op=False):
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For all_reduce, the input tensor must be tensor")
    if not isinstance(op, str):
        raise TypeError("For all_reduce, the input op type must be str")
    if op not in ('sum', 'prod', 'min', 'max'):
        raise TypeError(
            "For all_reduce, the input op value must be one of sum, prod, min, max")
    group = _get_group(group)
    tensor = _contiguous(tensor)
    all_reduce_op = _get_cache_prim(P.AllReduce)(op=op, group=group)
    output = all_reduce_op(tensor)
    return _deal_comm_outputs(output, async_op)


def bprop_pynative(input_x, weight, bias, running_mean, running_var, eps, momentum,
                   process_group, world_size, output, doutput):
    _, mean_param, invstd_param, count_all_param = output
    dout, _, _, _ = doutput

    # 不支持 KBK模式
    if not dout.is_contiguous():
        dout = dout.contiguous()

    grad_input = grad_weight = grad_bias = None

    inputG = True
    weightG = True
    biasG = True

    # calculate local stats as well as grad_weight / grad_bias
    sum_dy, sum_dy_xmu, grad_weight, grad_bias = batch_norm_reduce_grad(
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
        sum_dy_shape = shape(sum_dy)
        num_channels = sum_dy_shape[0]
        combined = mint.cat([sum_dy, sum_dy_xmu], dim=0)

        new_combined = all_reduce(combined, group=process_group)

        sum_dy, sum_dy_xmu = mint.split(new_combined, num_channels)

        # backward pass for gradient calculation
        grad_input = batch_norm_elemt_grad(
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


def bprop_kbk(input_x, weight, bias, running_mean, running_var, eps, momentum,
              process_group, world_size, output, doutput):
    _, mean_param, invstd_param, count_all_param = output
    dout, _, _, _ = doutput

    dout = dout.contiguous()

    grad_input = grad_weight = grad_bias = None

    inputG = True
    weightG = True
    biasG = True

    # calculate local stats as well as grad_weight / grad_bias
    sum_dy, sum_dy_xmu, grad_weight, grad_bias = batch_norm_reduce_grad(
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
        sum_dy_shape = shape(sum_dy)
        num_channels = sum_dy_shape[0]
        combined = mint.cat([sum_dy, sum_dy_xmu], dim=0)

        new_combined = all_reduce(combined, group=process_group)

        sum_dy, sum_dy_xmu = mint.split(new_combined, num_channels)

        # backward pass for gradient calculation
        grad_input = batch_norm_elemt_grad(
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


def construct_pynative(input, weight, bias, running_mean, running_var, eps, momentum, process_group,
                       world_size, self_num_features, self_world_size):
    if self_world_size != world_size:
        raise ValueError('World Size Error')
    if not input.is_contiguous():
        input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()

    input_shape = shape(input)
    input_numel = ops.numel(input)
    size = int(input_numel // input_shape[1])
    if size == 1 and world_size < 2:
        raise ValueError(
            'Expected more than 1 value per channel when training, got input size {}'.format(size))

    # calculate mean/invstd for input.
    mean, invstd = mint.batch_norm_stats(input, eps)
    count = mint.full((1,), input_numel //
                      input_shape[1], dtype=mean.dtype)

    num_channels = input_shape[1]
    if self_num_features != num_channels:
        raise ValueError('Features Error')
    # C, C, 1 -> (2C + 1)
    combined = mint.cat([mean, invstd, count], dim=0)
    # Use allgather instead of allreduce because count could be different across
    # ranks, simple all reduce op can not give correct results.
    # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
    # all gathered mean, invstd and count.
    # world_size * (2C + 1)
    combined = all_gather_into_tensor(combined, process_group)
    combined = ops.reshape(combined, [world_size, -1])
    # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
    mean_val_all, invstd_val_all, count_val_all = mint.split(
        combined, num_channels, dim=1)
    # calculate global mean & invstd
    mean, invstd = mint.batch_norm_gather_stats_with_counts(input, mean_val_all, invstd_val_all, running_mean,
                                                            running_var, momentum, eps, count_val_all.view(-1))

    # apply element-wise normalization
    out = mint.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
    return (out, mean, invstd, count_val_all.view(-1))


def construct_kbk(input, weight, bias, running_mean, running_var, eps, momentum, process_group,
                  world_size, self_num_features, self_world_size):
    if self_world_size != world_size:
        raise ValueError('World Size Error')
    input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()

    input_shape = shape(input)
    input_numel = ops.numel(input)
    size = int(input_numel // input_shape[1])
    if size == 1 and world_size < 2:
        raise ValueError(
            'Expected more than 1 value per channel when training, got input size {}'.format(size))

    # calculate mean/invstd for input.
    mean, invstd = mint.batch_norm_stats(input, eps)
    count = mint.full((1,), input_numel //
                      input_shape[1], dtype=mean.dtype)

    num_channels = input_shape[1]
    if self_num_features != num_channels:
        raise ValueError('Features Error')
    # C, C, 1 -> (2C + 1)
    combined = mint.cat([mean, invstd, count], dim=0)
    # Use allgather instead of allreduce because count could be different across
    # ranks, simple all reduce op can not give correct results.
    # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
    # all gathered mean, invstd and count.
    # world_size * (2C + 1)
    combined = all_gather_into_tensor(combined, process_group)
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


class SyncBatchNormInner(Cell):
    def __init__(self, self_num_features, self_world_size):
        super(SyncBatchNormInner, self).__init__()
        self.num_features = self_num_features
        self.world_size = self_world_size
        self.mode = context.get_context("mode")
        if self.mode == 1:
            self.fn_bprop = bprop_pynative
            self.fn_construct = construct_pynative
        else:
            self.fn_bprop = bprop_kbk
            self.fn_construct = construct_kbk

    def construct(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        return self.fn_construct(input, weight, bias, running_mean, running_var, eps, momentum, process_group,
                                 world_size, self.num_features, self.world_size)

    def bprop(self, input_x, weight, bias, running_mean, running_var, eps, momentum,
              process_group, world_size, output, doutput):
        return self.fn_bprop(input_x, weight, bias, running_mean, running_var, eps, momentum,
                             process_group, world_size, output, doutput)


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
