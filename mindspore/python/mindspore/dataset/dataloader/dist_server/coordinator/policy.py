import time
import collections
import io

class CoordinatorPolicy:
    def __init__(self):
        # 存储节点信息: {node_id: {'host': str, 'port': int, 'stats': dict}}
        self.nodes = {} 
        self.EMA_ALPHA = 0.2
        print("--- [Policy] Weighted + Latency-Aware Policy initialized. ---")

    def register_node(self, node_id, host, port, weight=1.0):
        """
        注册或更新节点信息
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "host": host,
                "port": port,
                "stats": {
                    "outstanding_requests": 0,
                    "weight": float(weight) if weight > 0 else 1.0,
                    "avg_latency": 0.1  
                }
            }
            print(f"--- [Policy] Node '{node_id}' registered ({host}:{port}) weight={weight}. ---")
        else:
            self.nodes[node_id]["host"] = host
            self.nodes[node_id]["port"] = port

    def assign_best_node(self, client_id, indices):
        """
        根据负载均衡算法选择最佳节点
        """
        if not self.nodes:
            return None

        best_node_id = None
        lowest_score = float('inf')

        for node_id, node_data in self.nodes.items():
            stats = node_data["stats"]
            
            weight = stats["weight"]
            outstanding = stats["outstanding_requests"]
            latency = stats["avg_latency"]
            score = ((outstanding / weight) + 1) * latency
            
            if score < lowest_score:
                lowest_score = score
                best_node_id = node_id

        if best_node_id:
            self.nodes[best_node_id]["stats"]["outstanding_requests"] += 1
            
            target_node = self.nodes[best_node_id]
            return {
                "server_node_id": best_node_id,
                "host": target_node["host"],
                "port": target_node["port"]
            }
        
        return None

    def report_completion(self, node_id, latency):
        """
        Client 汇报任务完成，更新节点延迟统计
        """
        if node_id in self.nodes:
            stats = self.nodes[node_id]["stats"]
            
            stats["outstanding_requests"] = max(0, stats["outstanding_requests"] - 1)
            
            current_avg = stats["avg_latency"]
            stats["avg_latency"] = (self.EMA_ALPHA * latency) + ((1 - self.EMA_ALPHA) * current_avg)