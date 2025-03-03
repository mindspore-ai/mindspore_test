# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""API for agent"""

from concurrent import futures
import grpc
import time
from ..protocol import agent_pb2, agent_pb2_grpc
import mindspore.log as logger
import json

class Agent:
    """
    Base class for dynamic networking nodes.

    """

    def __init__(self, worker_num, local_worker_num, sched_host, sched_port, node_rank):
        self.worker_num = worker_num
        self.local_worker_num = local_worker_num
        self.sched_host = sched_host
        self.sched_port = sched_port
        self.node_rank = node_rank


class AgentClient(Agent):
    def __init__(self, worker_num, local_worker_num, sched_host, sched_port, node_rank):
        super().__init__(worker_num, local_worker_num, sched_host, sched_port, node_rank)
        channel = grpc.insecure_channel(sched_host + ':' + str(sched_port))
        self.stub = agent_pb2_grpc.AgentRegServiceStub(channel)
        self.node_rank = node_rank
        self.local_worker_num = local_worker_num

    def register(self):
        request = agent_pb2.AgentRegReq()
        request.node_rank = self.node_rank
        request.local_worker_num = self.local_worker_num
        
        while True:
            try:
                response = self.stub.Register(request)
                logger.warning(f"Node rank {self.node_rank} register to master success {response.success}.")
                break
            except Exception:
                print(f"Retry...")
                time.sleep(1)
        
    def check_cluster_status(self):
        request = agent_pb2.CheckClusterStatusReq()
        request.node_rank = self.node_rank
        request.local_worker_num = self.local_worker_num
        
        ready = False
        while not ready:
            response = self.stub.CheckClusterStatus(request, timeout=9000)
            if response.ready:
                ready = True
                return response.assigned_rank_list
            else:
                time.sleep(0.1)


class AgentRequestHandler(agent_pb2_grpc.AgentRegServiceServicer):
    def __init__(self, worker_num):
        self.node_to_worker_num = {}
        self.node_to_rank_list = {}
        self.worker_num = worker_num
        self.ready = False

    def _accum_rank(self):
        begin_rank = 0
        for k, v in self.node_to_worker_num.items():
            end_rank = begin_rank + v
            self.node_to_rank_list[str(k)] = list(range(begin_rank, end_rank))
            begin_rank = end_rank
            
    def Register(self, request, context=None):
        response = agent_pb2.AgentRegRsp()
        response.success = True
        node_rank = request.node_rank
        local_worker_num = request.local_worker_num
        self.node_to_worker_num[str(node_rank)] = local_worker_num
        logger.warning(f"Receive register request from node rank {node_rank} with worker number {local_worker_num}")
        if self.node_to_rank_list == {} and sum(self.node_to_worker_num.values()) == self.worker_num:
            logger.warning(f"Enough worker registered: {self.node_to_worker_num}")
            self.ready = True
            self._accum_rank()
        return response
    

    def CheckClusterStatus(self, request, context=None):
        logger.warning("check status request")
        response = agent_pb2.CheckClusterStatusRsp()
        logger.warning("check status request2")
        node_rank = request.node_rank
        logger.warning("check status request3")
        local_worker_num = request.local_worker_num
        logger.warning("check status request4")
        if self.ready:
            logger.warning(f"check status request5 {self.node_to_rank_list}")
            response.ready = True
            logger.warning(f"check status request6 {self.node_to_rank_list[str(node_rank)]}")
            response.assigned_rank_list.extend(self.node_to_rank_list[str(node_rank)])
            logger.warning(f"Cluster is ready, node {node_rank} assigned rank list is {response.assigned_rank_list}")
        else:
            response.ready = False
        return response

class AgentServer(Agent):
    def __init__(self, worker_num, local_worker_num, sched_host, sched_port, node_rank):
        super().__init__(worker_num, local_worker_num, sched_host, sched_port, node_rank)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        agent_pb2_grpc.add_AgentRegServiceServicer_to_server(AgentRequestHandler(worker_num), server)
        server.add_insecure_port(sched_host + ':' + str(sched_port))
        server.start()
        self.server = server
    
    def join(self):
        self.server.wait_for_termination()