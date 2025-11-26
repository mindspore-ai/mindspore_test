import ray
import collections
from worker import DataWorkerActor 

@ray.remote
class ServeNodeActor:

    def __init__(self, node_id, dataset_params, num_workers_per_node, df_ref):
        self.node_id = node_id
        self.dataset_params = dataset_params
        

        print(f"--- [SERVENODE {node_id}] Received DataFrame, re-putting to object store... ---")
        self.df_ref_for_workers = ray.put(df_ref)
        print(f"--- [SERVENODE {node_id}] Re-put complete. ---")
        
        self.worker_pool = []
        for i in range(num_workers_per_node):
            print(f"--- [SERVENODE {node_id}] Launching Worker {i+1}/{num_workers_per_node}... ---")
            self.worker_pool.append(DataWorkerActor.remote(
                self.dataset_params, 
                self.df_ref_for_workers 
            ))          
        self.worker_index = 0
        print(f"--- [SERVENODE {node_id}] Initialized with {num_workers_per_node} workers. ---")


    def process_indices(self, indices_batch):
        SUB_BATCH_SIZE = 64
        object_refs = []
        
        for i in range(0,len(indices_batch),SUB_BATCH_SIZE):
            sub_indices = indices_batch[i : i + SUB_BATCH_SIZE]
            worker_actor = self.worker_pool[self.worker_index % len(self.worker_pool)]
            self.worker_index += 1
            object_refs.append(worker_actor.get_batch.remote(sub_indices))
                   
        '''
        for idx in indices_batch:
            worker_actor = self.worker_pool[self.worker_index % len(self.worker_pool)]
            object_refs.append(worker_actor.get_item.remote(idx))
            self.worker_index += 1
        '''
        return object_refs


    def get_worker_count(self):
        return len(self.worker_pool)