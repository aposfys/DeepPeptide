from typing import Dict
from torch.utils.tensorboard import SummaryWriter

def add_dict_to_writer(metrics: Dict[str, float], writer: SummaryWriter, global_step: int, prefix: str = ''):
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, global_step=global_step)

def compute_metrics(*args, **kwargs):
    raise NotImplementedError("compute_metrics is missing in this repo.")

def compute_crf_metrics(*args, **kwargs):
    raise NotImplementedError("compute_crf_metrics is missing in this repo.")
