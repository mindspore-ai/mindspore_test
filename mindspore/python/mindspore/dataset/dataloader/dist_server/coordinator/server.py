from dist_rpc.server_rpc import CoordinatorRPCServer
from .policy import CoordinatorPolicy
from dist_rpc.common import RPCMethod
import io

class CoordinatorService:
    def __init__(self):
        self.servers = [] # [{'host': '1.2.3.4', 'port': 9200}, ...]
        self.pointer = 0
        self.policy = CoordinatorPolicy()

    def handle_register(self, payload, ctx):
        """
        Payload: {'host': '...', 'port': 9200, ...}
        """
        host = ctx.client_address[0]
        node_id = payload.get("client_id")
        port = payload.get("port")
        weight = payload.get("weight", 1.0)

        self.policy.register_node(node_id, host, port, weight)
        
        return "OK"

    def handle_assign(self, payload, ctx):
        client_id = payload.get("client_id")
        indices = payload.get("indices")
        node = self.policy.assign_best_node(client_id, indices)
        
        if node is None:
           raise RuntimeError("No available server nodes")
        '''
        if not self.servers:
            raise RuntimeError("No ServerNodes available!")
        
        node = self.servers[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.servers)
        '''
        return node
    
    def handle_report_completion(self, payload, ctx):
        """
        Payload 格式: {'node_id': 'servenode_node_0', 'latency': 0.15}
        """
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