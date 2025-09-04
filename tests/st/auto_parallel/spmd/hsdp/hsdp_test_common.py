from typing import List, Tuple
import re
import mindspore.nn as nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

hsdp_network_ckpt_path: str = "hsdp_network.ckpt"

def extract_metrics_from_log(log_path: str) -> List[Tuple]:
    # expected log format
    pattern = r"step: (\d+), loss: ([\d.]+)"
    # [(loss_0, ), (loss_1, )...(loss_n, )]
    indicator_list: List[Tuple] = []
    with open(log_path, mode="r", encoding="utf-8") as log:
        for line in log:
            effect_log_start_idx: int = line.find("step:")
            if effect_log_start_idx != -1:
                line = line[effect_log_start_idx:]
            else:
                continue
            match = re.match(pattern, line.strip("."))
            if match:
                loss = float(match.group(2))
                indicator_list.append((loss,))
    return indicator_list

class ErrorComparator:
    def __init__(self, baseline_log_path: str, hsdp_case_log_path: str):
        self.baseline_path = baseline_log_path
        self.hsdp_case_path = hsdp_case_log_path
        self.baseline_metrics = extract_metrics_from_log(self.baseline_path)
        self.hsdp_case_metrics = extract_metrics_from_log(self.hsdp_case_path)
        assert self.baseline_metrics,\
            f"For HSDP precision case, {self.baseline_path} hasn't capture any metrics, Please check the log."
        assert len(self.baseline_metrics) == len(self.hsdp_case_metrics),\
            f"For HSDP precision case, get {len(self.baseline_metrics)} steps from {self.baseline_path},\
                but get {len(self.hsdp_case_metrics)} steps from {self.hsdp_case_path}"

    def get_relative_absolute_error(self) -> float:
        baseline_loss_list = self._extract_loss_from_metrics(self.baseline_metrics)
        hsdp_case_loss_list = self._extract_loss_from_metrics(self.hsdp_case_metrics)
        total_step: int = len(baseline_loss_list)
        total_relative_abs_loss: float = 0.0
        for baseline_loss, hsdp_case_loss in zip(baseline_loss_list, hsdp_case_loss_list):
            total_relative_abs_loss += abs(hsdp_case_loss - baseline_loss) / baseline_loss
        return total_relative_abs_loss / total_step

    def get_first_step_rel_abs_error(self) -> bool:
        baseline_first_step_metric: Tuple = self.baseline_metrics[0]
        case_first_step_metric: Tuple = self.hsdp_case_metrics[0]
        rel_abs_err = abs(baseline_first_step_metric[0] - case_first_step_metric[0]) / baseline_first_step_metric[0]
        return rel_abs_err

    def get_rel_abs_error_of_steps(self) -> List[float]:
        baseline_loss_list = self._extract_loss_from_metrics(self.baseline_metrics)
        hsdp_case_loss_list = self._extract_loss_from_metrics(self.hsdp_case_metrics)
        errors: List[float] = []
        for baseline_loss, hsdp_case_loss in zip(baseline_loss_list, hsdp_case_loss_list):
            errors.append(abs(hsdp_case_loss - baseline_loss) / baseline_loss)
        return errors

    def _extract_loss_from_metrics(self, metrics_list: List[Tuple]):
        if metrics_list is None:
            return []
        loss_list: List[float] = [per_step_metric[0] for per_step_metric in metrics_list]
        return loss_list
