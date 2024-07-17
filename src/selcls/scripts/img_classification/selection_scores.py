
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
from tqdm import tqdm
from ...datasets import load_cifar_dataset
from ...models.utils import load_img_classification_model
from ...models.selectors import MSPSelector, EntropySelector, GiniSelector, RelUSelector, MSPCalSelector
from ...utils import save_yaml

import torch
from torch.utils.data import DataLoader


def run_model_on_dataset(model, dataset, selector, input_perturbation, temperature, batch_size=128, device="cuda:0"):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    
    device = torch.device(device)
    model = model.to(device)
    selector = selector.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    indices, logits, targets = [], [], []
    for idx, x, y in tqdm(dataloader, total=len(dataloader)):
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
        indices.append(idx.cpu())
    logits = torch.cat(logits, dim=0)
    targets = torch.cat(targets, dim=0)
    indices = torch.cat(indices, dim=0).numpy().astype(int)
    model = model.cpu()
    selector = selector.cpu()
    return indices, logits, targets


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
    output_dir: str = "./",
    device: str = "cuda:0",
    batch_size: int = 128,
    **kwargs,
):
    # Load train data and train the selector
    train_idx = pd.read_csv(train_list, header=None).values.flatten()
    df_logits = pd.read_csv(logits, index_col=0, header=0)
    df_targets = pd.read_csv(targets, index_col=0, header=0)
    train_logits = torch.from_numpy(df_logits.loc[train_idx,:].values).float()
    train_targets = torch.from_numpy(df_targets.loc[train_idx,:].values.flatten()).long()
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
    if temperature != 1.0 or input_perturbation != 0.0:
        model = load_img_classification_model(model, dataset, train_method, checkpoints_dir, seed)
        dataset = load_cifar_dataset(dataset, train_method, data_dir)
        indices, logits, targets = run_model_on_dataset(model, dataset, selector, input_perturbation, temperature, batch_size, device)
        if train_method == "openmix":
            logits = logits[:, :-1]
    else:
        logits, targets = torch.from_numpy(df_logits.values).float(), torch.from_numpy(df_targets.values.flatten()).long()
        indices = df_logits.index.values
    
    # Compute the scores with perturbed logits
    with torch.inference_mode():
        scores = selector.compute_score(logits / temperature)

    # Save results
    output_dir = Path(output_dir)
    pd.DataFrame(logits.numpy().astype(float), columns=df_logits.columns, index=indices).to_csv(output_dir / "logits.csv", index=True, header=True)
    pd.DataFrame(targets.numpy().astype(int), columns=df_targets.columns, index=indices).to_csv(output_dir / "targets.csv", index=True, header=True)
    pd.DataFrame(scores.numpy().astype(float), columns=["score"], index=indices).to_csv(output_dir / "scores.csv", index=True, header=True)
    save_yaml(selector.hparams, output_dir / "hparams.yaml")
    torch.save(selector.state_dict(), output_dir / "state_dict.pth")

    


    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)