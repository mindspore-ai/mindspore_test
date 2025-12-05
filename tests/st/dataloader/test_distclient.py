"""End-to-end tests for dist_client/dist_rpc helper modules."""

from __future__ import annotations

import contextlib
import importlib
import io
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = PROJECT_ROOT / "mindspore" / "python"
if str(PYTHON_SRC) not in sys.path:
	sys.path.insert(0, str(PYTHON_SRC))

from mindspore import Tensor

_dataset_mod = importlib.import_module("mindspore.dataset.dataloader.dist_client.dataset")
DummyTextDataset = _dataset_mod.DummyTextDataset
build_dummy_mapping_dataset = _dataset_mod.build_dummy_mapping_dataset

_loader_mod = importlib.import_module("mindspore.dataset.dataloader.dist_client.distributed_dataloader")
DistributedDataLoader = _loader_mod.DistributedDataLoader

_rpc_adapter_mod = importlib.import_module("mindspore.dataset.dataloader.dist_client.rpc_adapter")
_demo_run = _rpc_adapter_mod._demo_run
build_inmemory_rpc = _rpc_adapter_mod.build_inmemory_rpc
configure_rpc_clients = _rpc_adapter_mod.configure_rpc_clients

rpc_example = importlib.import_module("mindspore.dataset.dataloader.dist_rpc.example")


def _find_free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[1]


def test_dummy_text_dataset_shapes() -> None:
	dataset = DummyTextDataset(size=3, seq_length=6)
	assert len(dataset) == 3

	sample = dataset[0]
	assert isinstance(sample, list)
	assert sample and set(sample[0].keys()) == {"input_ids", "attention_mask", "labels"}
	assert sample[0]["input_ids"].shape == (6,)


def test_mapping_dataset_transform_applies_callable() -> None:
	def add_flag(record):
		record[0]["transformed"] = True
		return record

	dataset = build_dummy_mapping_dataset(transform=add_flag)
	sample = dataset[0]
	assert sample[0].get("transformed") is True


def test_distributed_dataloader_inmemory_microbatches() -> None:
	coordinator_factory, server_factory = build_inmemory_rpc()
	configure_rpc_clients(coordinator_factory, server_factory)

	dataset = DummyTextDataset(size=8, seq_length=4)
	loader = DistributedDataLoader(dataset, batch_size=4, microbatch_size=2, shuffle=False)

	microbatches = next(iter(loader))
	assert len(microbatches) == 2
	first_microbatch = microbatches[0]
	assert isinstance(first_microbatch, Tensor)
	assert first_microbatch.shape == (2, 2)


def test_dist_rpc_example_run() -> None:
	coord_port = _find_free_port()
	server_port = _find_free_port()
	buffer = io.StringIO()

	with contextlib.redirect_stdout(buffer):
		rpc_example.run_example(coordinator_port=coord_port, server_port=server_port)

	output = buffer.getvalue()
	assert "Assignment:" in output
	assert "Fetch result tensor" in output


def test_rpc_adapter_demo_run() -> None:
	coord_port = _find_free_port()
	server_port = _find_free_port()
	buffer = io.StringIO()

	with contextlib.redirect_stdout(buffer):
		_demo_run(coordinator_port=coord_port, server_port=server_port, timeout=1.0)

	output = buffer.getvalue()
	assert "Demo assignment" in output
