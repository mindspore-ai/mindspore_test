# services/data_node/server.py
import argparse
import pandas as pd
import torch
import io
import time
from concurrent.futures import ThreadPoolExecutor

from rpc.server_rpc import ServerNodeRPCServer
from rpc.client_rpc import CoordinatorRPCClient
from services.data_node.processor import DataProcessor

# 全局处理器
processor = None
executor = ThreadPoolExecutor(max_workers=4) # 简单的并发池

def handle_fetch(payload, ctx):
    """
    Payload 格式: {'indices': [1, 2, 3], 'client_id': '...'}
    """
    indices = payload.get("indices", [])
    if not indices:
        return None

    future = executor.submit(processor.get_batch, indices)
    result_dict = future.result() # 阻塞等待结果

    if result_dict is None:
        return None

    # !!! 接口适配关键点 !!!
    # 原有的 RPC 协议可能期望返回 Tensor。
    # 为了支持多模态 (Image, Text, Mask)，我们将整个字典视为一个 Object 保存。
    # 只要 rpc/server_rpc.py 里的 _encode_response_payload 使用了 TorchSerializer
    # 且 rpc/serde.py 使用了 torch.save，那么保存字典是支持的。
    # 注意：你需要确认 rpc/client_rpc.py 里的类型检查是否通过。
    buffer = io.BytesIO()
    torch.save(result_dict, buffer)
    return buffer.getvalue() # 返回 bytes 类型


def register_to_coordinator(coordinator_host, coordinator_port, local_port, node_id="node_0"):
    """
    循环尝试向 Coordinator 注册，直到成功
    """
    client = CoordinatorRPCClient(coordinator_host, coordinator_port)
    while True:
        try:
            print(f"Attempting to register to Coordinator at {coordinator_host}:{coordinator_port}...")
            # 调用我们在第一步中新增的方法
            success = client.register_servernode(node_id=node_id, port=local_port)
            if success:
                print(">>> Registration SUCCESS!")
                break
        except Exception as e:
            print(f"Registration failed ({e}), retrying in 2s...")
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--coord_host", type=str, default="127.0.0.1") 
    parser.add_argument("--coord_port", type=int, default=9100)        
    parser.add_argument("--node_id", type=str, default="node_0")       
    parser.add_argument("--metadata", type=str, default="/home/lyh/ray_temp/metadata.parquet")
    parser.add_argument("--tokenizer", type=str, default="/home/lyh/ray_temp/models--openai--clip-vit-base-patch32")
    args = parser.parse_args()

    # 1. 加载数据
    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_parquet(args.metadata)
    
    # 2. 初始化处理器
    global processor
    processor = DataProcessor(args.tokenizer, df)

    # 3. 启动 RPC Server
    server = ServerNodeRPCServer("0.0.0.0", args.port)
    server.on_fetch(handle_fetch)
    
    import threading
    t = threading.Thread(target=register_to_coordinator, args=(
        args.coord_host, 
        args.coord_port, 
        args.port,
        args.node_id
    ), daemon=True)
    t.start()
    
    
    print(f"ServerNode running on port {args.port}...")
    server.serve_forever()

if __name__ == "__main__":
    main()