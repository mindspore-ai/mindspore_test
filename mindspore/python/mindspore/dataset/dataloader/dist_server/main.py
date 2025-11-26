import ray
import time
import torch
import pandas as pd
import os
import sys

from coordinator import CoordinatorActor
from servenode import ServeNodeActor   
from client import ClientActor 


METADATA_PATH = "/home/lyh/ray_temp/metadata.parquet" 
LOCAL_TOKENIZER_PATH = "/home/lyh/ray_temp/models--openai--clip-vit-base-patch32"

def main():
    # --- (配置) ---
    NUM_GPUS = 8
    
    NODE_CONFIGS = [
        {"id": "node_0", "num_cpus": 8}, 
        {"id": "node_1", "num_cpus": 8},
        {"id": "node_2", "num_cpus": 8},
        {"id": "node_3", "num_cpus": 8}  
    ]

    TOTAL_SAMPLES = 0
    BATCH_SIZE = 256
    NUM_BATCHES_PER_CLIENT =50

    # --- 1. 检查和加载元数据 ---
    print(f"--- [MAIN] 检查 Tokenizer 路径: {LOCAL_TOKENIZER_PATH} ---")
    if not os.path.isdir(LOCAL_TOKENIZER_PATH):
        print(f"!!! [MAIN] 错误: Tokenizer 路径未找到 !!!", file=sys.stderr)
        return

    print(f"--- [MAIN] 检查 Metadata 路径: {METADATA_PATH} ---")
    try:
        df = pd.read_parquet(METADATA_PATH)
    except FileNotFoundError:
        print(f"!!! [MAIN] 错误: Metadata file '{METADATA_PATH}' not found.", file=sys.stderr)
        return
        
    TOTAL_SAMPLES = len(df)
    print(f"--- [MAIN] Metadata loaded. Found {TOTAL_SAMPLES} samples. ---")

    DATASET_PARAMS = {
        "tokenizer_path": LOCAL_TOKENIZER_PATH
    }
    
    # --- 2. 启动 Ray ---
    total_cpus_needed = sum(c["num_cpus"] for c in NODE_CONFIGS) + NUM_GPUS + 2
    ray.init(
        num_cpus=total_cpus_needed, 
        resources={"npu":NUM_GPUS}, 
        local_mode=False
    )
    print("--- [MAIN] Ray cluster started (local_mode=False) ---")

    print("--- [MAIN] Putting DataFrame into Ray object store... ---")
    df_ref = ray.put(df)
    print("--- [MAIN] DataFrame put complete. ---")

    # --- 3. 创建 Actors ---
    coordinator = CoordinatorActor.options(name="coordinator").remote(
        total_samples=TOTAL_SAMPLES,
        batch_size=BATCH_SIZE
    )

    registration_tasks = []
    servenode_handles = []
    for config in NODE_CONFIGS:
        node_id = config["id"]
        num_workers = config["num_cpus"]
        
        servenode = ServeNodeActor.options(name=f"servenode_{node_id}").remote(
            node_id=node_id,
            dataset_params=DATASET_PARAMS, 
            num_workers_per_node=num_workers,
            df_ref=df_ref
        )
        servenode_handles.append(servenode)
        
        worker_count_ref = servenode.get_worker_count.remote()
        worker_count = ray.get(worker_count_ref)
        
        reg_ref = coordinator.register_servenode.remote(
            f"servenode_{node_id}", 
            servenode,
            worker_count
        )
        registration_tasks.append(reg_ref)

    print(f"--- [MAIN] Waiting for all {len(registration_tasks)} ServeNodes to register... ---")
    ray.get(registration_tasks)
    print(f"--- [MAIN] All ServeNodes registered successfully. ---")

    # --- 4. 启动客户端 ---
    print(f"--- [MAIN] Creating {NUM_GPUS} ClientActors... ---")
    client_tasks = []
    coordinator_handle = ray.get_actor("coordinator")
    
    
    client_actors = []
    for i in range(NUM_GPUS):
        actor = ClientActor.remote(
            client_id=i,
            coordinator_handle=coordinator_handle
        )
        client_actors.append(actor)
        
    print(f"--- [MAIN] Triggering training loop on all clients... ---")
    for actor in client_actors:
        task_ref = actor.run_training_loop.remote(
            num_batches=NUM_BATCHES_PER_CLIENT
        )
        client_tasks.append(task_ref)

    # --- 5. 等待完成  ---
    start_time = time.time()
    results = ray.get(client_tasks)
    end_time = time.time()
    print("--- [MAIN] All clients finished. ---")

    # --- 6. 打印 ---
    # ...
    total_batches_processed = NUM_GPUS * NUM_BATCHES_PER_CLIENT
    total_duration = end_time - start_time
    throughput = total_batches_processed / total_duration

    print("\n--- Summary (V3 RealData) ---")
    print(f"Total batches processed: {total_batches_processed}")
    print(f"Total execution time: {total_duration:.2f} seconds")
    print(f"Overall throughput: {throughput:.2f} batches/second")
    
    ray.shutdown()

if __name__ == "__main__":
    main()