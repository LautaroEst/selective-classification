from functools import partial
from pathlib import Path
from typing import Literal
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from ...datasets import load_cifar_dataset
from ...models.utils import load_img_classification_model

@torch.inference_mode()
def run_model_on_dataset(model, dataset, batch_size=128, device="cuda:0"):
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits, targets = [], []
    for x, y in tqdm(dataloader, total=len(dataloader)):
        x, y = x.to(device), y.to(device)
        batch_logits = model(x)
        logits.append(batch_logits.cpu())
        targets.append(y.cpu())
    logits = torch.cat(logits, dim=0).numpy().astype(float)
    targets = torch.cat(targets, dim=0).numpy().astype(int)
    return logits, targets


def main(
    model: Literal["densenet121", "resnet34"],
    dataset: Literal["cifar10", "cifar100"],
    train_method: Literal["ce", "logit_norm", "mixup", "openmix", "regmixup"],
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 0,
    logits_output: str = "logits.csv",
    targets_output: str = "targets.csv",
    device: str = "cuda:0",
    batch_size: int = 128,
):

    # Load model and dataset    
    model = load_img_classification_model(model, dataset, train_method, checkpoints_dir, seed)
    dataset = load_cifar_dataset(dataset, train_method, data_dir)

    # Run model on dataset
    logits, targets = run_model_on_dataset(model, dataset, batch_size, device)
    if train_method == "openmix":
        logits = logits[:, :-1]

    # Save results
    logits = pd.DataFrame(logits, columns=dataset.classes, index=np.arange(len(dataset)))
    logits.to_csv(logits_output, index=True, header=True)
    targets = pd.DataFrame(targets, columns=["target"], index=np.arange(len(dataset)))
    targets.to_csv(targets_output, index=True, header=True)
    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)