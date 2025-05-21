import pytest
import numpy as np
import mindspore
import mindspore.context as context
from mindspore.mint.nn.layer import SyncBatchNorm
from mindspore import Tensor
from mindspore.communication import init, create_group, get_local_rank

init()


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_sync_batch_norm_forward_world_size_2_channel_2_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)

    sync_batch_norm = SyncBatchNorm(
        num_features=2, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train(False)
    input_x = Tensor(np.linspace(0, 5, 2*2*2*2),
                     mindspore.float32).reshape(2, 2, 2, 2)
    output_data = sync_batch_norm(input_x)

    expect_output_data = np.array([[[[0.0000, 0.3333],
                                     [0.6667, 1.0000]],

                                    [[1.3333, 1.6667],
                                     [2.0000, 2.3333]]],

                                   [[[2.6667, 3.0000],
                                     [3.3333, 3.6666]],

                                    [[4.0000, 4.3333],
                                     [4.6666, 5.0000]]]])
    expect_running_mean = np.array([0., 0.])
    expect_running_var = np.array([1., 1.])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)

    sync_batch_norm = SyncBatchNorm(
        num_features=2, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()
    output_data = sync_batch_norm(input_x)
    expect_output_data = np.array([[[[-1.3242, -1.0835],
                                     [-0.8427, -0.6019]],

                                    [[-1.3242, -1.0835],
                                     [-0.8427, -0.6019]]],

                                   [[[0.6019, 0.8427],
                                     [1.0835, 1.3242]],

                                    [[0.6019, 0.8427],
                                     [1.0835, 1.3242]]]])
    expect_running_mean = np.array([0.1833, 0.3167])
    expect_running_var = np.array([1.1044, 1.1044])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_forward_world_size_2_channel_2_dim_4_diff_process_group(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    local_rank = get_local_rank()
    if local_rank in [0, 1]:
        group01 = "0-1"
        rank_ids01 = [0, 1]
        create_group(group01, rank_ids01)

        sync_batch_norm = SyncBatchNorm(
            num_features=2, process_group=group01, dtype=mindspore.float32)
        sync_batch_norm.set_train()
        input_x = Tensor(np.linspace(0, 5, 2*2*2*2),
                         mindspore.float32).reshape(2, 2, 2, 2)
        output_data = sync_batch_norm(input_x)
        expect_output_data = np.array([[[[-1.3242, -1.0835],
                                         [-0.8427, -0.6019]],

                                        [[-1.3242, -1.0835],
                                         [-0.8427, -0.6019]]],

                                       [[[0.6019, 0.8427],
                                         [1.0835, 1.3242]],

                                        [[0.6019, 0.8427],
                                         [1.0835, 1.3242]]]])
        expect_running_mean = np.array([0.1833, 0.3167])
        expect_running_var = np.array([1.1044, 1.1044])
        assert np.allclose(output_data.asnumpy(),
                           expect_output_data, rtol=0.005, atol=0.005)
        assert np.allclose(sync_batch_norm.running_mean.asnumpy(
        ), expect_running_mean, rtol=0.005, atol=0.005)
        assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                           expect_running_var, rtol=0.005, atol=0.005)
    else:
        group23 = "2-3"
        rank_ids23 = [2, 3]
        create_group(group23, rank_ids23)

        sync_batch_norm = SyncBatchNorm(
            num_features=2, process_group=group23, dtype=mindspore.float32)
        sync_batch_norm.set_train()
        input_x = Tensor(np.linspace(0, 10, 2*2*2*2),
                         mindspore.float32).reshape(2, 2, 2, 2)
        output_data = sync_batch_norm(input_x)
        expect_output_data = np.array([[[[-1.3242, -1.0835],
                                         [-0.8427, -0.6019]],

                                        [[-1.3242, -1.0835],
                                         [-0.8427, -0.6019]]],

                                       [[[0.6019, 0.8427],
                                         [1.0835, 1.3242]],

                                        [[0.6019, 0.8427],
                                         [1.0835, 1.3242]]]])
        expect_running_mean = np.array([0.3667, 0.6333])
        expect_running_var = np.array([1.7178, 1.7178])
        assert np.allclose(output_data.asnumpy(),
                           expect_output_data, rtol=0.005, atol=0.005)
        assert np.allclose(sync_batch_norm.running_mean.asnumpy(
        ), expect_running_mean, rtol=0.005, atol=0.005)
        assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                           expect_running_var, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_forward_world_size_3_channel_3_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train(False)
    input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                     mindspore.float32).reshape(3, 3, 3, 3)
    output_data = sync_batch_norm(input_x)

    expect_output_data = np.array([[[[0.0000, 0.0625, 0.1250],
                                     [0.1875, 0.2500, 0.3125],
                                     [0.3750, 0.4375, 0.5000]],

                                    [[0.5625, 0.6250, 0.6875],
                                     [0.7500, 0.8125, 0.8750],
                                     [0.9375, 1.0000, 1.0625]],

                                    [[1.1250, 1.1875, 1.2500],
                                     [1.3125, 1.3750, 1.4375],
                                     [1.5000, 1.5625, 1.6250]]],


                                   [[[1.6875, 1.7500, 1.8125],
                                     [1.8750, 1.9375, 2.0000],
                                     [2.0625, 2.1250, 2.1875]],

                                    [[2.2500, 2.3125, 2.3750],
                                     [2.4375, 2.5000, 2.5625],
                                     [2.6250, 2.6875, 2.7500]],

                                    [[2.8125, 2.8750, 2.9375],
                                     [3.0000, 3.0625, 3.1250],
                                     [3.1875, 3.2500, 3.3125]]],


                                   [[[3.3750, 3.4375, 3.5000],
                                     [3.5625, 3.6250, 3.6875],
                                     [3.7500, 3.8125, 3.8750]],

                                    [[3.9375, 4.0000, 4.0625],
                                     [4.1250, 4.1875, 4.2500],
                                     [4.3125, 4.3750, 4.4375]],

                                    [[4.5000, 4.5625, 4.6250],
                                     [4.6875, 4.7500, 4.8125],
                                     [4.8750, 4.9375, 5.0000]]]])
    expect_running_mean = np.array([0., 0., 0.])
    expect_running_var = np.array([1., 1., 1.])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()
    output_data = sync_batch_norm(input_x)
    expect_output_data = np.array([[[[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]]],


                                   [[[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, -8.5931e-08, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]]],


                                   [[[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]]]])
    expect_running_mean = np.array([0.1938, 0.2500, 0.3063])
    expect_running_var = np.array([1.0949, 1.0949, 1.0949])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_forward_world_size_3_channel_3_dim_4_affine_False(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, affine=False, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train(False)
    input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                     mindspore.float32).reshape(3, 3, 3, 3)
    output_data = sync_batch_norm(input_x)

    expect_output_data = np.array([[[[0.0000, 0.0625, 0.1250],
                                     [0.1875, 0.2500, 0.3125],
                                     [0.3750, 0.4375, 0.5000]],

                                    [[0.5625, 0.6250, 0.6875],
                                     [0.7500, 0.8125, 0.8750],
                                     [0.9375, 1.0000, 1.0625]],

                                    [[1.1250, 1.1875, 1.2500],
                                     [1.3125, 1.3750, 1.4375],
                                     [1.5000, 1.5625, 1.6250]]],


                                   [[[1.6875, 1.7500, 1.8125],
                                     [1.8750, 1.9375, 2.0000],
                                     [2.0625, 2.1250, 2.1875]],

                                    [[2.2500, 2.3125, 2.3750],
                                     [2.4375, 2.5000, 2.5625],
                                     [2.6250, 2.6875, 2.7500]],

                                    [[2.8125, 2.8750, 2.9375],
                                     [3.0000, 3.0625, 3.1250],
                                     [3.1875, 3.2500, 3.3125]]],


                                   [[[3.3750, 3.4375, 3.5000],
                                     [3.5625, 3.6250, 3.6875],
                                     [3.7500, 3.8125, 3.8750]],

                                    [[3.9375, 4.0000, 4.0625],
                                     [4.1250, 4.1875, 4.2500],
                                     [4.3125, 4.3750, 4.4375]],

                                    [[4.5000, 4.5625, 4.6250],
                                     [4.6875, 4.7500, 4.8125],
                                     [4.8750, 4.9375, 5.0000]]]])
    expect_running_mean = np.array([0., 0., 0.])
    expect_running_var = np.array([1., 1., 1.])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, affine=False, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()
    output_data = sync_batch_norm(input_x)
    expect_output_data = np.array([[[[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]]],


                                   [[[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, -8.5931e-08, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]]],


                                   [[[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]]]])
    expect_running_mean = np.array([0.1938, 0.2500, 0.3063])
    expect_running_var = np.array([1.0949, 1.0949, 1.0949])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_mean.asnumpy(),
                       expect_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(sync_batch_norm.running_var.asnumpy(),
                       expect_running_var, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_forward_world_size_3_channel_3_dim_4_track_running_stats_False(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1-2"
    rank_ids = [0, 1, 2]
    create_group(group, rank_ids)

    input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                     mindspore.float32).reshape(3, 3, 3, 3)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, track_running_stats=False, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()
    output_data = sync_batch_norm(input_x)
    expect_output_data = np.array([[[[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]],

                                    [[-1.3966e+00, -1.3516e+00, -1.3065e+00],
                                     [-1.2615e+00, -1.2164e+00, -1.1714e+00],
                                     [-1.1263e+00, -1.0813e+00, -1.0362e+00]]],


                                   [[[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, -8.5931e-08, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]],

                                    [[-1.8021e-01, -1.3516e-01, -9.0106e-02],
                                     [-4.5053e-02, 0.0000e+00, 4.5053e-02],
                                     [9.0106e-02, 1.3516e-01, 1.8021e-01]]],


                                   [[[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]],

                                    [[1.0362e+00, 1.0813e+00, 1.1263e+00],
                                     [1.1714e+00, 1.2164e+00, 1.2615e+00],
                                     [1.3065e+00, 1.3516e+00, 1.3966e+00]]]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_forward_world_size_2_channel_3_dim_4_diff_nhw(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()
    if get_local_rank() == 0:
        input_x = Tensor(np.linspace(0, 5, 2*3*2*2),
                         mindspore.float32).reshape(2, 3, 2, 2)
    else:
        input_x = Tensor(np.linspace(0, 5, 3*3*3*3),
                         mindspore.float32).reshape(3, 3, 3, 3)
    output_data = sync_batch_norm(input_x)

    if get_local_rank() == 0:
        expect_output_data = np.array([[[[-1.3534, -1.1958],
                                         [-1.0383, -0.8807]],

                                        [[-1.1869, -1.0287],
                                         [-0.8704, -0.7121]],

                                        [[-1.0100, -0.8525],
                                         [-0.6949, -0.5373]]],


                                       [[[0.5373, 0.6949],
                                         [0.8525, 1.0100]],

                                        [[0.7121, 0.8704],
                                         [1.0287, 1.1869]],

                                        [[0.8807, 1.0383],
                                         [1.1958, 1.3534]]]])
    else:
        expect_output_data = np.array([[[[-1.3534, -1.3081, -1.2628],
                                         [-1.2175, -1.1722, -1.1269],
                                         [-1.0816, -1.0363, -0.9910]],

                                        [[-1.4104, -1.3650, -1.3195],
                                         [-1.2740, -1.2285, -1.1830],
                                         [-1.1375, -1.0920, -1.0465]],

                                        [[-1.4551, -1.4098, -1.3645],
                                         [-1.3192, -1.2739, -1.2287],
                                         [-1.1834, -1.1381, -1.0928]]],


                                       [[[-0.1303, -0.0850, -0.0397],
                                         [0.0056, 0.0509, 0.0962],
                                         [0.1415, 0.1868, 0.2321]],

                                        [[-0.1820, -0.1365, -0.0910],
                                         [-0.0455, 0.0000, 0.0455],
                                         [0.0910, 0.1365, 0.1820]],

                                        [[-0.2321, -0.1868, -0.1415],
                                         [-0.0962, -0.0509, -0.0056],
                                         [0.0397, 0.0850, 0.1303]]],


                                       [[[1.0928, 1.1381, 1.1834],
                                         [1.2287, 1.2739, 1.3192],
                                         [1.3645, 1.4098, 1.4551]],

                                        [[1.0465, 1.0920, 1.1375],
                                         [1.1830, 1.2285, 1.2740],
                                         [1.3195, 1.3650, 1.4104]],

                                        [[0.9910, 1.0363, 1.0816],
                                         [1.1269, 1.1722, 1.2175],
                                         [1.2628, 1.3081, 1.3534]]]])
    assert np.allclose(output_data.asnumpy(),
                       expect_output_data, rtol=0.005, atol=0.005)


@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_sync_batch_norm_backward_world_size_2_channel_3_dim_4(mode):
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    group = "0-1"
    rank_ids = [0, 1]
    create_group(group, rank_ids)

    sync_batch_norm = SyncBatchNorm(
        num_features=3, process_group=group, dtype=mindspore.float32)
    sync_batch_norm.set_train()

    input_x = Tensor(np.linspace(0, 5, 2*3*2*2),
                     mindspore.float32).reshape(2, 3, 2, 2)
    grad_net = mindspore.grad(sync_batch_norm, grad_position=(0,))
    gradient_input_x = grad_net(input_x)

    expect_grap_input_x = np.array([[[[-8.3214e-08, -7.2118e-08],
                                      [-6.1023e-08, -4.9928e-08]],

                                     [[4.1607e-08, 3.6059e-08],
                                      [3.0512e-08, 2.4964e-08]],

                                     [[8.3214e-08, 7.2118e-08],
                                      [6.1023e-08, 4.9928e-08]]],


                                    [[[4.9928e-08, 6.1023e-08],
                                      [7.2118e-08, 8.3214e-08]],

                                     [[-2.4964e-08, -3.0512e-08],
                                      [-3.6059e-08, -4.1607e-08]],

                                     [[-4.9928e-08, -6.1023e-08],
                                      [-7.2118e-08, -8.3214e-08]]]])
    assert np.allclose(gradient_input_x.asnumpy(),
                       expect_grap_input_x, rtol=0.005, atol=0.005)
