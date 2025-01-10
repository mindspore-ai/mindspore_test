import numpy as np
import mindspore.ops.operations as P
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.nn import Cell, Momentum
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from tests.st.auto_parallel.optimizer_parallel import FakeData


class ParallelConv2DNet(Cell):
    def __init__(self, mul_size, mul2_size, weight_size, out_channel, strategy=None):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        mul2_np = np.full(mul2_size, 0.5, dtype=np.float32)
        weight_np = np.full(weight_size, 1, dtype=np.float32)
        self.weight = Parameter(Tensor(weight_np), name="weight")
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.mul2_weight = Parameter(Tensor(mul2_np), name="mul2_weight")
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=2, stride=1, pad_mode="same"
        )
        if strategy is not None:
            self.conv2d.shard(strategy)

    def construct(self, inputs, label):
        x = self.mul(inputs, self.mul_weight)
        x = self.conv2d(x, self.weight)
        x = self.mul2(x, self.mul2_weight)
        return x


def test_auto_parallel_fault_error_comm_order():
    '''
    Feature: Runtime fault test case.
    Description: Test different workers have inconsistent execution order of communication operators.
    Expectation: Run success
    '''
    net = ParallelConv2DNet(
        mul_size=(128, 2, 2, 1024),
        mul2_size=(128, 10, 2, 1024),
        weight_size=(10, 2, 2, 2),
        out_channel=10,
        strategy=((4, 1, 1, 2), (1, 1, 1, 1)),
    )
    dataset = FakeData(size=256, batch_size=16, image_size=(2, 2, 1024), use_parallel=True)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=8)
    opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
    model = Model(network=net, optimizer=opt)
    model.train(epoch=2, train_dataset=dataset)
