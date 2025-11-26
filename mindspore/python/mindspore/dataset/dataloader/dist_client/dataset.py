from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset



# dataset 需要负责定义好 __len__ 和 __getitem__ 
# 通常是用户定义好的接口
# 用户端在卸载执行过程中需要知道 dataset 的一些元数据
# 如果是真实的数据集路径，load 的时候就可以进行分析
class DummyTextDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]



class MappingDataset(Dataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]




def build_dummy_dataset(task_type: str, size: int, max_seq_len: int) -> "Dataset":
    if task_type == "text":
        return DummyTextDataset(size=size, seq_length=max_seq_len)
    else:
        raise ValueError(f"Dummy dataset type ({task_type}) is not supported.")



def build_dummy_mapping_dataset(
    transform: Optional[Callable] = None,
) -> "Dataset":
    """
    Build a mapping dataset using dummy data.
    
    Args:
        data_path (str): Ignored for dummy dataset, but kept for API compatibility.
        transform (Optional[Callable]): Transform function to apply to samples.
        source_name (Optional[str]): Source name to pass to transform if applicable.
        
    Returns:
        Dataset: A MappingDataset wrapping a DummyTextDataset.
    """
    # Default parameters for the dummy dataset
    size = 1000
    seq_length = 128
    
    dataset = DummyTextDataset(size=size, seq_length=seq_length)

    return MappingDataset(data=dataset, transform=transform)


if __name__ == "__main__":
    print("Testing DummyTextDataset...")
    dataset = DummyTextDataset(size=5, seq_length=10)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    # DummyTextDataset returns a list of dicts
    print("Sample keys:", sample[0].keys())
    print("Input IDs shape:", sample[0]["input_ids"].shape)

    print("\nTesting build_dummy_mapping_dataset...")
    
    def simple_transform(sample):
        # Example transform: just add a key
        # sample is List[Dict]
        sample[0]["transformed"] = True
        return sample

    mapping_dataset = build_dummy_mapping_dataset(transform=simple_transform)
    print(f"Mapping dataset length: {len(mapping_dataset)}")
    mapped_sample = mapping_dataset[0]
    print("Mapped sample keys:", mapped_sample[0].keys())
    assert mapped_sample[0].get("transformed") is True
    print("Transform applied successfully.")



