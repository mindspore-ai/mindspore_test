# services/coordinator/server.py
from rpc.server_rpc import CoordinatorRPCServer
from .policy import CoordinatorPolicy
from rpc.common import RPCMethod
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
        # 注意：这里需要一种机制获取 ServerNode 的真实 IP，
        # 如果 Payload 里没传，可以从 ctx.client_address 获取
        '''
        node_info = {
            "host": payload.get("host") or ctx.client_address[0],
            "port": payload.get("port")
        }
        self.servers.append(node_info)
        print(f"[Coordinator] Registered node: {node_info}")
        '''
        host = ctx.client_address[0]
        node_id = payload.get("client_id")
        port = payload.get("port")
        weight = payload.get("weight", 1.0)

        # 2. 调用 Policy 进行注册
        self.policy.register_node(node_id, host, port, weight)
        
        return "OK"

    def handle_assign(self, payload, ctx):
        """
        简单轮询调度
        """
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
        # 返回分配结果，必须包含 host 和 port 以供 Client 连接
        return node
    
    def handle_report_completion(self, payload, ctx):
        """
        处理 Client 上报的完成信息
        Payload 格式: {'node_id': 'servenode_node_0', 'latency': 0.15}
        """
        node_id = payload.get("node_id")
        latency = payload.get("latency")

        if node_id is None or latency is None:
            print("[Coordinator] Warning: Invalid completion report received.")
            return "Invalid Payload"

        # 调用 Policy 更新节点状态 (EMA 延迟计算 & outstanding 计数减一)
        self.policy.report_completion(node_id, float(latency))
        
        # 可选: 打印调试日志
        # print(f"[Coordinator] Report: Node {node_id} finished in {latency:.4f}s")
        
        return "OK"

def main():
    service = CoordinatorService()
    server = CoordinatorRPCServer("0.0.0.0", 9100)
    
    server.on_register_client(lambda p, c: "SessionOK") # 简单 mock
    server.on_assign_servernode(service.handle_assign)
    server.register_handler(RPCMethod.REPORT_COMPLETION, service.handle_report_completion)
    server.register_handler(RPCMethod.REGISTER_SERVERNODE, service.handle_register)
    server.register_handler(RPCMethod.REPORT_COMPLETION, service.handle_report_completion)

    print("Coordinator running on port 9100...")
    server.serve_forever()

if __name__ == "__main__":
    main()