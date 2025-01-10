import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import ReduceOp
from mindspore.communication import init
from mindspore.communication.management import get_rank
from mindspore.communication.comm_func import send, recv


class MsNet(nn.Cell):
    def __init__(self, rank):
        super(MsNet, self).__init__()
        self.rank = rank
        self.matmul = ops.MatMul()
        self.all_reduce = ops.AllReduce(ReduceOp.SUM)
        self.all_gather = ops.AllGather()
        self.reduce_scatter = ops.ReduceScatter(ReduceOp.SUM)

    def ms_all_reduce(self, x, y, z):
        out = self.matmul(x, y)
        out = self.all_reduce(out)
        out = self.matmul(out, z)
        out = self.all_reduce(out)
        return out

    def ms_all_gather(self, x, y, z):
        out = self.matmul(x, y)
        out = self.all_gather(out)
        out = self.matmul(out, z)
        out = self.all_gather(out)
        return out

    def ms_reduce_scatter(self, x, y, z):
        out = self.matmul(x, y)
        out = self.reduce_scatter(out)
        out = self.matmul(out, z)
        out = self.reduce_scatter(out)
        return out

    def ms_send(self, z, dst, sr_tag):
        send(z, dst=dst, group="hccl_world_group", tag=sr_tag)

    def ms_recv(self, z, src, sr_tag):
        out = recv(z, src=src, group="hccl_world_group", tag=sr_tag)
        return out

    def ms_send_recv(self, z, rank, src_dst_rank, sr_tag):
        if rank < 4:
            self.ms_send(z, dst=src_dst_rank, sr_tag=sr_tag)
            out = self.ms_recv(z, src=src_dst_rank, sr_tag=sr_tag)
        else:
            out = self.ms_recv(z, src=src_dst_rank, sr_tag=sr_tag)
            self.ms_send(z, dst=src_dst_rank, sr_tag=sr_tag)
        return out

class NpNet():
    def __init__(self, rank, global_rank=8):
        self.rank = rank
        self.global_rank = global_rank

    def matmul_np(self, input_np=None):
        matmul_list = []
        if input_np is None:
            for rank_i in range(self.global_rank):
                x_np_rank_i = rank_i * np.ones([64, 4]).astype(np.float32)
                y_np_rank_i = rank_i * np.ones([4, 8]).astype(np.float32)
                out_mat_mul = np.matmul(x_np_rank_i, y_np_rank_i)
                matmul_list.append(out_mat_mul)
        else:
            for rank_i in range(self.global_rank):
                z_np_rank_i = rank_i * np.ones([8, 2]).astype(np.float32)
                out_mat_mul_2 = np.matmul(input_np, z_np_rank_i)
                matmul_list.append(out_mat_mul_2)
        return matmul_list

    def np_all_reduce(self):
        out = self.matmul_np()
        out = np.sum(out, axis=0)
        out = self.matmul_np(input_np=out)
        out = np.sum(out, axis=0)
        return out

    def np_all_gather(self):
        out = self.matmul_np()
        out = np.concatenate(out)
        out = self.matmul_np(input_np=out)
        out = np.concatenate(out)
        return out

    def np_reduce_scatter(self):
        out = self.matmul_np()
        out = np.split(np.sum(out, axis=0), self.global_rank, axis=0)[self.rank]
        out = self.matmul_np(input_np=out)
        out = np.split(np.sum(out, axis=0), self.global_rank, axis=0)[self.rank]
        return out

    def np_send_recv(self, src_dst_rank):
        out = src_dst_rank * np.ones([8, 2]).astype(np.float32)
        return out

if __name__ == '__main__':
    init()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    rank_id = get_rank()
    net_ms = MsNet(rank=rank_id)
    net_np = NpNet(rank=rank_id, global_rank=8)

    x_np = rank_id * np.ones([64, 4]).astype(np.float32)
    y_np = rank_id * np.ones([4, 8]).astype(np.float32)
    z_np = rank_id * np.ones([8, 2]).astype(np.float32)
    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    z_ms = Tensor(z_np)

    out_list = []
    # test cross cluster AllReduce
    out_all_reduce_np = net_np.np_all_reduce()
    out_all_reduce_ms = net_ms.ms_all_reduce(x_ms, y_ms, z_ms)
    out_list.append([out_all_reduce_np, out_all_reduce_ms])

    # test cross cluster AllGather
    out_all_gather_np = net_np.np_all_gather()
    out_all_gather_ms = net_ms.ms_all_gather(x_ms, y_ms, z_ms)
    out_list.append([out_all_gather_np, out_all_gather_ms])

    # test cross cluster ReduceScatter
    out_reduce_scatter_np = net_np.np_reduce_scatter()
    out_reduce_scatter_ms = net_ms.ms_reduce_scatter(x_ms, y_ms, z_ms)
    out_list.append([out_reduce_scatter_np, out_reduce_scatter_ms])

    # test cross cluster Send and Receive
    # set the send and recv map, the format is {rank_id: {'src_dst_rank_id': src_dst_rank_id, 'tag': tag}}
    send_recv_map = {0: {'src_dst_rank_id': 4, 'tag': 0}, 4: {'src_dst_rank_id': 0, 'tag': 0},
                     1: {'src_dst_rank_id': 5, 'tag': 1}, 5: {'src_dst_rank_id': 1, 'tag': 1},
                     2: {'src_dst_rank_id': 6, 'tag': 2}, 6: {'src_dst_rank_id': 2, 'tag': 2},
                     3: {'src_dst_rank_id': 7, 'tag': 3}, 7: {'src_dst_rank_id': 3, 'tag': 3}}
    src_dst_rank_id = send_recv_map[rank_id]['src_dst_rank_id']
    tag = send_recv_map[rank_id]['tag']
    out_send_recv_np = net_np.np_send_recv(src_dst_rank_id)
    out_send_recv_ms = net_ms.ms_send_recv(z_ms, rank_id, src_dst_rank_id, tag)
    out_list.append([out_send_recv_np, out_send_recv_ms])

    # check test result
    cross_cluster_ops_list = ["AllReduce", "AllGather", "ReduceScatter", "Send and Receive"]
    all_passed = True
    for i in range(len(cross_cluster_ops_list)):
        out_np = out_list[i][0]
        out_ms = out_list[i][1]
        if np.allclose(out_ms.asnumpy(), out_np):
            print(f"test cross cluster {cross_cluster_ops_list[i]} success, the ms output is equal to the np output")
        else:
            print(f"test cross cluster {cross_cluster_ops_list[i]} failed, the ms output is not equal to the np output")
            all_passed = False
            break
    if all_passed:
        print("test all cross cluster communication operators success")
    else:
        print("test some cross cluster communication operators failed, please check the log")
