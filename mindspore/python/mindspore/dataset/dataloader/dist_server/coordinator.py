import ray
import random
import time
import collections
#dddddd
@ray.remote
class CoordinatorActor:
    
    def __init__(self, total_samples, batch_size):
        self.global_indices = list(range(total_samples))
        random.shuffle(self.global_indices)
        self.current_index = 0
        self.batch_size = batch_size
        
        self.servenode_stats = {} 
        self.servenode_names = []
        
        self.EMA_ALPHA = 0.2
        print("--- [COORDINATOR] Actor V3 (Weighted + Latency-Aware) initialized. ---")

    
    def register_servenode(self, name, handle, weight):
        if name not in self.servenode_stats:
            self.servenode_stats[name] = {
                "handle": handle,
                "outstanding_requests": 0,
                "weight": float(weight) if weight > 0 else 1.0,
                "avg_latency": 0.1  # (初始默认 100ms)
            }
            self.servenode_names.append(name)
            print(f"--- [COORDINATOR] ServeNode '{name}' registered with weight {weight}. ---")


    def request_batch(self):
        indices_to_process = self._get_next_indices()
        if not indices_to_process:
            return None, None
            
        if not self.servenode_names:
            print("--- [COORDINATOR] No ServeNodes registered. Waiting... ---")
            time.sleep(1)
            return None, None

        # 调度逻辑
        best_node_name = None
        lowest_score = float('inf')

        for name in self.servenode_names:
            stats = self.servenode_stats[name]
            weight = stats["weight"]
            outstanding = stats["outstanding_requests"]
            latency = stats["avg_latency"]
            
            score = ((outstanding / weight) + 1) * latency
            
            if score < lowest_score:
                lowest_score = score
                best_node_name = name
        '''       
        best_node_name = min(
            self.servenode_stats.keys(),
            key=lambda name: self.servenode_stats[name]["outstanding_requests"]
        )
        ''' 

        self.servenode_stats[best_node_name]["outstanding_requests"] += 1       
        target_servenode = self.servenode_stats[best_node_name]["handle"]
        batch_refs_promise = target_servenode.process_indices.remote(indices_to_process)
        
        return batch_refs_promise, best_node_name


    def report_batch_completion(self, node_name, latency):
        """
        报告完成
        """
        if node_name in self.servenode_stats:
            stats = self.servenode_stats[node_name]
            
            stats["outstanding_requests"] = max(0, stats["outstanding_requests"] - 1)
            
            current_avg = stats["avg_latency"]
            stats["avg_latency"] = (self.EMA_ALPHA * latency) + ((1 - self.EMA_ALPHA) * current_avg)


    def _get_next_indices(self):
        start = self.current_index
        end = start + self.batch_size
        if start >= len(self.global_indices):
            return None       
        self.current_index = end
        return self.global_indices[start:end]