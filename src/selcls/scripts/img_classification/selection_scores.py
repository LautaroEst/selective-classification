
import pandas as pd
import numpy as np
from typing import Literal
from tqdm import tqdm
from ...datasets import load_cifar_dataset
from ...models.utils import load_img_classification_model
from ...models.selectors import MSPSelector, EntropySelector, GiniSelector, RelUSelector, MSPCalSelector

import torch
from torch.utils.data import DataLoader


def run_model_on_dataset(model, dataset, selector, input_perturbation, temperature, batch_size=128, device="cuda:0"):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    
    device = torch.device(device)
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits, targets = [], []
    for x, y in tqdm(dataloader, total=len(dataloader)):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        batch_logits = model(x)
        batch_scores = selector.compute_score(batch_logits / temperature)
        batch_scores.sum().backward()
        x = x - input_perturbation * torch.sign(-x.grad)
        x = x.detach()
        with torch.inference_mode():
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
    input_perturbation: float,
    temperature: float,
    score: Literal["msp", "entropy", "gini", "relu", "mspcal-ts", "mspcal-dp"],
    logits: str,
    targets: str,
    train_list: str,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 0,
    logits_output: str = "logits.csv",
    targets_output: str = "targets.csv",
    device: str = "cuda:0",
    batch_size: int = 128,
    **kwargs,
):
    # Load train data and train the selector
    train_idx = pd.read_csv(train_list, header=None).values.flatten()
    train_logits = torch.from_numpy(pd.read_csv(logits, index_col=0, header=0).loc[train_idx,:].values).float()
    train_targets = torch.from_numpy(pd.read_csv(targets, index_col=0, header=0).loc[train_idx,:].values.flatten()).long()
    n_classes = train_logits.size(1)
    if score == "msp":
        selector = MSPSelector(n_classes, random_state=seed)
    elif score == "entropy":
        selector = EntropySelector(n_classes, random_state=seed)
    elif score == "gini":
        selector = GiniSelector(n_classes, random_state=seed)
    elif score == "relu":
        selector = RelUSelector(n_classes, random_state=seed, **kwargs)
    elif score == "mspcal-ts":
        selector = MSPCalSelector("ts", n_classes, random_state=seed, **kwargs)
    elif score == "mspcal-dp":
        selector = MSPCalSelector("dp", n_classes, random_state=seed, **kwargs)
    else:
        raise ValueError(f"Unknown score function: {score}")
    selector.fit(train_logits, train_targets)

    # Run model on dataset with perturbation and temperature
    model = load_img_classification_model(model, dataset, train_method, checkpoints_dir, seed)
    dataset = load_cifar_dataset(dataset, train_method, data_dir)
    logits, targets = run_model_on_dataset(model, dataset, selector, input_perturbation, temperature, batch_size, device)
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