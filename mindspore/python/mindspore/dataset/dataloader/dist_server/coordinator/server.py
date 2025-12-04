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
# ==============================================================================
"""
RPC server for coordinator
"""


from dist_rpc.server_rpc import CoordinatorRPCServer
from .policy import CoordinatorPolicy
from dist_rpc.common import RPCMethod

class CoordinatorService:
    def __init__(self):
        self.servers = [] 
        self.pointer = 0
        self.policy = CoordinatorPolicy()

    def handle_register(self, payload, ctx):
        """
        Payload: {'node_id': '...','host': '...', 'port': 9200, 'latency': int}
        """
        host = ctx.client_address[0]
        node_id = payload.get("client_id")
        port = payload.get("port")
        weight = payload.get("weight", 1.0)

        self.policy.register_node(node_id, host, port, weight)
        
        return "OK"

    def handle_assign(self, payload):
        client_id = payload.get("client_id")
        indices = payload.get("indices")
        node = self.policy.assign_best_node(client_id, indices)
        
        if node is None:
           raise RuntimeError("No available server nodes")
        return node
    
    def handle_report_completion(self, payload):
        node_id = payload.get("node_id")
        latency = payload.get("latency")

        if node_id is None or latency is None:
            print("[Coordinator] Warning: Invalid completion report received.")
            return "Invalid Payload"
        self.policy.report_completion(node_id, float(latency))
        
        return "OK"

def main():
    service = CoordinatorService()
    server = CoordinatorRPCServer("0.0.0.0", 9100)
    
    server.on_register_client(lambda p, c: "SessionOK") 
    server.on_assign_servernode(service.handle_assign)
    server.register_handler(RPCMethod.REPORT_COMPLETION, service.handle_report_completion)
    server.register_handler(RPCMethod.REGISTER_SERVERNODE, service.handle_register)
    server.register_handler(RPCMethod.REPORT_COMPLETION, service.handle_report_completion)

    print("Coordinator running on port 9100...")
    server.serve_forever()

if __name__ == "__main__":
    main()