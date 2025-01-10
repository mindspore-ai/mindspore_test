import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()


class Network(nn.Cell):
    def __init__(self):
        super()._init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(28 * 28, 10, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        logits = self.relu(x)
        return logits

def forward_fn(net, loss_fn, data, label):
    logits = net(data)
    losses = loss_fn(logits, label)
    return losses, logits

if __name__ == "__main__":
    net_work = Network()

    loss_func = nn.CrossEntropyLoss()
    optimizer = nn.SGD(net_work.parameters(), 1e-2)

    grad_fn = ms.value_and_grad(loss_func, None, net_work.trainable_parameters(), has_aux=True)
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

    train_data = Tensor(np.random.rand(32, 28, 28), dtype=ms.float32)
    train_label = Tensor(np.random.rand(32, 28, 1), dtype=ms.float32)

    for epoch in range(10):
        (loss, _), grads = grad_fn(train_data, train_label)
        grads = grad_reducer(grads)
        optimizer(grads)
