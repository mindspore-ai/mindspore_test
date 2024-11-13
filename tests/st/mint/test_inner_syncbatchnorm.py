import pytest
import numpy as np
import mindspore
import mindspore.context as context
from mindspore.mint.nn.layer._functions import _SyncBatchNorm
from mindspore import Tensor
from mindspore import Parameter
from mindspore.communication import init, create_group, get_local_rank, get_group_size

init()


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_forward_world_size_2_channel_2_dim_2(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(2, 2)
    input_x = Tensor([[1.0, 2.0], [3.0, 4.0]], mindspore.float32)
    running_mean = Parameter(
        Tensor([0.5, 1.5], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    output_data = sync_batch_norm(
        input_x, weight, bias, running_mean, running_var, eps, momentum, group, world_size)
    expect_output_data = np.array([[-3.0000, -3.0000],
                                   [1.0000, 1.0000]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_forward_world_size_2_channel_2_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(2, 2)
    input_x = Tensor(np.linspace(0, 5, 2*2*2*2),
                     mindspore.float32).reshape(2, 2, 2, 2)
    running_mean = Parameter(
        Tensor([0.5, 1.5], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    output_data = sync_batch_norm(
        input_x, weight, bias, running_mean, running_var, eps, momentum, group, world_size)
    expect_output_data = np.array([[[[-3.6485, -3.1669],
                                     [-2.6854, -2.2039]],
                                    [[-3.6485, -3.1669],
                                     [-2.6854, -2.2039]]],
                                   [[[0.2039, 0.6854],
                                     [1.1669, 1.6485]],
                                    [[0.2039, 0.6854],
                                     [1.1669, 1.6485]]]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_forward_world_size_3_channel_3_dim_2(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(3, 3)
    input_x = Tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], mindspore.float32)
    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="runnig_var")
    weight = Tensor([2.0, 2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    output_data = sync_batch_norm(
        input_x, weight, bias, running_mean, running_var, eps, momentum, group, world_size)
    expect_output_data = np.array([[-3.0000, -3.0000, -3.0000],
                                   [1.0000, 1.0000, 1.0000]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_forward_world_size_3_channel_3_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(3, 3)
    input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                     mindspore.float32).reshape(3, 3, 3, 3)
    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    output_data = sync_batch_norm(
        input_x, weight, bias, running_mean, running_var, eps, momentum, group, world_size)
    expect_output_data = np.array([[[[-3.7933, -3.7032, -3.6131],
                                     [-3.5230, -3.4329, -3.3427],
                                     [-3.2526, -3.1625, -3.0724]],

                                    [[-3.7933, -3.7032, -3.6131],
                                     [-3.5230, -3.4329, -3.3427],
                                     [-3.2526, -3.1625, -3.0724]],

                                    [[-3.7933, -3.7032, -3.6131],
                                     [-3.5230, -3.4329, -3.3427],
                                     [-3.2526, -3.1625, -3.0724]]],


                                   [[[-1.3604, -1.2703, -1.1802],
                                     [-1.0901, -1.0000, -0.9099],
                                     [-0.8198, -0.7297, -0.6396]],

                                    [[-1.3604, -1.2703, -1.1802],
                                     [-1.0901, -1.0000, -0.9099],
                                     [-0.8198, -0.7297, -0.6396]],

                                    [[-1.3604, -1.2703, -1.1802],
                                     [-1.0901, -1.0000, -0.9099],
                                     [-0.8198, -0.7297, -0.6396]]],


                                   [[[1.0724, 1.1625, 1.2526],
                                     [1.3427, 1.4329, 1.5230],
                                     [1.6131, 1.7032, 1.7933]],

                                    [[1.0724, 1.1625, 1.2526],
                                     [1.3427, 1.4329, 1.5230],
                                     [1.6131, 1.7032, 1.7933]],

                                    [[1.0724, 1.1625, 1.2526],
                                     [1.3427, 1.4329, 1.5230],
                                     [1.6131, 1.7032, 1.7933]]]])
    expect_running_mean = np.array([0.6438, 1.6000, 2.1063])
    expect_running_var = np.array([0.2849, 0.3749, 1.9949])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_forward_world_size_2_channel_3_dim_4_diff_nhw(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(3, 2)
    if get_local_rank() == 0:
        input_x = Tensor(np.linspace(0, 5, 2*3*2*2),
                         mindspore.float32).reshape(2, 3, 2, 2)
    else:
        input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                         mindspore.float32).reshape(3, 3, 3, 3)

    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    output_data = sync_batch_norm(
        input_x, weight, bias, running_mean, running_var, eps, momentum, group, world_size)
    if get_local_rank() == 0:
        expect_output_data = np.array([[[[-3.7068, -3.3917],
                                         [-3.0766, -2.7614]],

                                        [[-3.3738, -3.0573],
                                         [-2.7408, -2.4243]],

                                        [[-3.0201, -2.7049],
                                         [-2.3898, -2.0747]]],


                                       [[[0.0747, 0.3898],
                                         [0.7049, 1.0201]],

                                        [[0.4243, 0.7408],
                                         [1.0573, 1.3738]],

                                        [[0.7614, 1.0766],
                                         [1.3917, 1.7068]]]])
    else:
        expect_output_data = np.array([[[[-3.7068, -3.6162, -3.5256],
                                         [-3.4350, -3.3444, -3.2538],
                                         [-3.1632, -3.0726, -2.9820]],

                                        [[-3.8209, -3.7299, -3.6389],
                                         [-3.5479, -3.4569, -3.3659],
                                         [-3.2749, -3.1839, -3.0929]],

                                        [[-3.9103, -3.8197, -3.7291],
                                         [-3.6385, -3.5479, -3.4573],
                                         [-3.3667, -3.2761, -3.1855]]],


                                       [[[-1.2607, -1.1701, -1.0795],
                                         [-0.9889, -0.8983, -0.8077],
                                         [-0.7171, -0.6265, -0.5359]],

                                        [[-1.3640, -1.2730, -1.1820],
                                         [-1.0910, -1.0000, -0.9090],
                                         [-0.8180, -0.7270, -0.6360]],

                                        [[-1.4641, -1.3735, -1.2829],
                                         [-1.1923, -1.1017, -1.0111],
                                         [-0.9205, -0.8299, -0.7393]]],


                                       [[[1.1855, 1.2761, 1.3667],
                                         [1.4573, 1.5479, 1.6385],
                                         [1.7291, 1.8197, 1.9103]],

                                        [[1.0929, 1.1839, 1.2749],
                                         [1.3659, 1.4569, 1.5479],
                                         [1.6389, 1.7299, 1.8209]],

                                        [[0.9820, 1.0726, 1.1632],
                                         [1.2538, 1.3444, 1.4350],
                                         [1.5256, 1.6162, 1.7068]]]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_backward_world_size_2_channel_3_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})

    input_x = Tensor(np.linspace(0, 5, 2*3*2*2),
                     mindspore.float32).reshape(2, 3, 2, 2)
    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="running_var")
    weight = Parameter(
        Tensor([2.0, 2.0, 2.0], mindspore.float32), name="weight")
    bias = Parameter(
        Tensor([-1.0, -1.0, -1.0], mindspore.float32), name="bias")

    momentum = 0.1
    eps = 1e-5

    sync_batch_norm = _SyncBatchNorm(3, 2)
    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0, 1, 2))

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    grad_input_x, grad_weight, grad_bias = grad_net(input_x, weight, bias, running_mean, running_var, eps, momentum,
                                                    group, world_size)
    expect_grad_input_x = np.array([[[[0., 0.],
                                      [0., 0.]],

                                     [[0., 0.],
                                      [0., 0.]],

                                     [[0., 0.],
                                      [0., 0.]]],


                                    [[[0., 0.],
                                      [0., 0.]],

                                     [[0., 0.],
                                      [0., 0.]],

                                     [[0., 0.],
                                      [0., 0.]]]])
    expect_grad_weight = np.array([0., 0., 0.])
    expect_grad_bias = np.array([8., 8., 8.])
    assert np.allclose(grad_input_x.asnumpy(),
                       expect_grad_input_x, rtol=0.005, atol=0.005)
    assert np.allclose(grad_weight.asnumpy(),
                       expect_grad_weight, rtol=0.005, atol=0.005)
    assert np.allclose(grad_bias.asnumpy(), expect_grad_bias,
                       rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_backward_world_size_2_channel_2_dim_2(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(2, 2)
    input_x = Tensor([[1.0, 2.0], [3.0, 4.0]], mindspore.float32)
    running_mean = Parameter(
        Tensor([0.5, 1.5], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0, 1, 2))
    gradient = grad_net(input_x, weight, bias, running_mean,
                        running_var, eps, momentum, group, world_size)
    expect_grad_input = np.array([[0, 0], [0, 0]])
    expect_grad_weight = np.array([[0, 0]])
    expect_grad_bias = np.array([[2, 2]])
    assert np.allclose(gradient[0].asnumpy(),
                       expect_grad_input, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[1].asnumpy(),
                       expect_grad_weight, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[2].asnumpy(),
                       expect_grad_bias, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_backward_world_size_2_channel_2_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(2, 2)
    input_x = Tensor(np.linspace(0, 5, 2*2*2*2),
                     mindspore.float32).reshape(2, 2, 2, 2)
    running_mean = Parameter(
        Tensor([0.5, 1.5], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0, 1, 2))
    gradient = grad_net(input_x, weight, bias, running_mean,
                        running_var, eps, momentum, group, world_size)
    expect_grad_input = np.array([[[[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]]],
                                  [[[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]]]])
    expect_grad_weight = np.array([0., 0.])
    expect_grad_bias = np.array([8., 8.])
    assert np.allclose(gradient[0].asnumpy(),
                       expect_grad_input, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[1].asnumpy(),
                       expect_grad_weight, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[2].asnumpy(),
                       expect_grad_bias, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_backward_world_size_3_channel_3_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(3, 3)
    input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                     mindspore.float32).reshape(3, 3, 3, 3)
    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0, 1, 2))
    gradient = grad_net(input_x, weight, bias, running_mean,
                        running_var, eps, momentum, group, world_size)
    expect_grad_input = np.array([[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                  [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                  [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]])
    expect_grad_weight = np.array([0., 0., 0.])
    expect_grad_bias = np.array([27., 27., 27.])
    assert np.allclose(gradient[0].asnumpy(),
                       expect_grad_input, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[1].asnumpy(),
                       expect_grad_weight, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[2].asnumpy(),
                       expect_grad_bias, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_inner_sync_batch_norm_backward_world_size_2_channel_3_dim_4_diff_nhw(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    sync_batch_norm = _SyncBatchNorm(3, 2)
    if get_local_rank() == 0:
        input_x = Tensor(np.linspace(0, 5, 2*3*2*2),
                         mindspore.float32).reshape(2, 3, 2, 2)
    else:
        input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                         mindspore.float32).reshape(3, 3, 3, 3)

    running_mean = Parameter(
        Tensor([0.5, 1.5, 2.0], mindspore.float32), name="running_mean")
    running_var = Parameter(
        Tensor([0.1, 0.2, 2.0], mindspore.float32), name="running_var")
    weight = Tensor([2.0, 2.0, 2.0], mindspore.float32)
    bias = Tensor([-1.0, -1.0, -1.0], mindspore.float32)

    momentum = 0.1
    eps = 1e-5

    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)
    world_size = get_group_size(group)

    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0, 1, 2))
    gradient = grad_net(input_x, weight, bias, running_mean,
                        running_var, eps, momentum, group, world_size)

    if get_local_rank() == 0:
        expect_grad_input = np.array([[[[-1.9372e-07, -1.7117e-07],
                                        [-1.4862e-07, -1.2606e-07]],
                                       [[0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00]],
                                       [[-1.1566e-07, -9.7616e-08],
                                        [-7.9574e-08, -6.1531e-08]]],
                                      [[[7.6914e-08, 9.9467e-08],
                                        [1.2202e-07, 1.4457e-07]],
                                       [[0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00]],
                                       [[1.0085e-07, 1.1889e-07],
                                        [1.3694e-07, 1.5498e-07]]]])
        expect_grad_weight = np.array([-1.3735, 0.0000, 1.3735])
        expect_grad_bias = np.array([8., 8., 8.])
    else:
        expect_grad_input = np.array([[[[-1.9372e-07, -1.8724e-07, -1.8076e-07],
                                        [-1.7427e-07, -1.6779e-07, -1.6130e-07],
                                        [-1.5482e-07, -1.4834e-07, -1.4185e-07]],
                                       [[0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00]],
                                       [[-1.6663e-07, -1.6144e-07, -1.5625e-07],
                                        [-1.5107e-07, -1.4588e-07, -1.4069e-07],
                                        [-1.3551e-07, -1.3032e-07, -1.2513e-07]]],
                                      [[[-1.8655e-08, -1.2171e-08, -5.6866e-09],
                                        [7.9741e-10, 7.2814e-09, 1.3765e-08],
                                        [2.0249e-08, 2.6734e-08, 3.3218e-08]],
                                       [[0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00]],
                                       [[-2.6574e-08, -2.1387e-08, -1.6200e-08],
                                        [-1.1012e-08, -5.8252e-09, -6.3794e-10],
                                        [4.5493e-09, 9.7365e-09, 1.4924e-08]]],
                                      [[[1.5641e-07, 1.6290e-07, 1.6938e-07],
                                        [1.7587e-07, 1.8235e-07, 1.8883e-07],
                                        [1.9532e-07, 2.0180e-07, 2.0829e-07]],
                                       [[0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00]],
                                       [[1.1348e-07, 1.1867e-07, 1.2386e-07],
                                        [1.2904e-07, 1.3423e-07, 1.3942e-07],
                                        [1.4460e-07, 1.4979e-07, 1.5498e-07]]]])
        expect_grad_weight = np.array([1.3735, 0.0000, -1.3735])
        expect_grad_bias = np.array([27., 27., 27.])

    assert np.allclose(gradient[0].asnumpy(),
                       expect_grad_input, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[1].asnumpy(),
                       expect_grad_weight, rtol=0.005, atol=0.005)
    assert np.allclose(gradient[2].asnumpy(),
                       expect_grad_bias, rtol=0.005, atol=0.005)
