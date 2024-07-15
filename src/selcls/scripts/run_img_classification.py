from pathlib import Path
from typing import Literal
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from ..models import DenseNet121Small, ResNet34
from ..datasets import CIFAR10, CIFAR100


def load_img_classification_model(model: str, dataset: str, train_method: str, checkpoints_dir: str, seed: int):

    checkpoint_path = Path(checkpoints_dir) / train_method / f"{model}_{dataset}" / str(seed)
    if (checkpoint_path / "best.pth").exists():
        checkpoint_path = checkpoint_path / "best.pth"
    elif (checkpoint_path / "last.pt").exists():
        checkpoint_path = checkpoint_path / "last.pt"
    else:
        raise FileNotFoundError(f"Checkpoint not found in {checkpoint_path}")

    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100

    if model == "densenet121":
        model = DenseNet121Small(num_classes)
    elif model == "resnet34":
        model = ResNet34(num_classes)
    
    w = torch.load(checkpoint_path, map_location="cpu")
    w = {k.replace("module.", ""): v for k, v in w.items()}
    if train_method == "openmix":
        # add one class to model output
        model._modules[list(model._modules.keys())[-1]] = torch.nn.Linear(
            model._modules[list(model._modules.keys())[-1]].in_features,
            model._modules[list(model._modules.keys())[-1]].out_features + 1,
        )
    model.load_state_dict(w)

    return model
    
    
def load_cifar_dataset(dataset: str, train_method: str, data_dir: str):
    if dataset == "cifar10":
        dataset = CIFAR10(data_dir, split="test", train_method=train_method, download=True)
    elif dataset == "cifar100":
        dataset = CIFAR100(data_dir, split="test", train_method=train_method, download=True)
    return dataset


def run_model_on_dataset(model, dataset, input_perturbation, temperature, batch_size=128, device="cuda:0"):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
        
    device = torch.device(device)
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits, targets = [], []
    for x, y in tqdm(dataloader, total=len(dataloader)):
        x, y = x.to(device), y.to(device)
        if input_perturbation > 0.0:
            x.requires_grad_(True)
            batch_logits = model(x)
            if temperature != 1.0:
                batch_logits /= temperature
            

            
            x_perturbed = x + input_perturbation * torch.sign()
        logits.append(y_pred.cpu())
        targets.append(y.cpu())
    logits = torch.cat(logits, dim=0).numpy().astype(float)
    targets = torch.cat(targets, dim=0).numpy().astype(int)
    return logits, targets


def main(
    model: Literal["densenet121", "resnet34"],
    dataset: Literal["cifar10", "cifar100"],
    train_method: Literal["ce", "logit_norm", "mixup", "openmix", "regmixup"],
    input_perturbation: float = 0.0,
    temperature: float = 1.0,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 0,
    output: str = "output.csv",
    device: str = "cuda:0",
    batch_size: int = 128,
):

    # Load model and dataset    
    model = load_img_classification_model(model, dataset, train_method, checkpoints_dir, seed)
    dataset = load_cifar_dataset(dataset, train_method, data_dir)

    # Run model on dataset
    logits, targets = run_model_on_dataset(model, dataset, input_perturbation, temperature, batch_size, device)
    if train_method == "openmix":
        logits = logits[:, :-1]
    outputs = pd.DataFrame(logits, columns=dataset.classes, index=np.arange(len(dataset)))
    outputs["target"] = targets
    outputs.to_csv(output, index=True, header=True)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)