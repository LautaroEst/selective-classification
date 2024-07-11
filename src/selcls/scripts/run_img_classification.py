
import torch 

from pathlib import Path
from typing import Literal
from ..models import DenseNet121Small, ResNet34
from ..datasets import CIFAR10, CIFAR100


def load_model(model: str, dataset: str, train_method: str, checkpoints_dir: str, seed: int):

    checkpoint_path = Path(checkpoints_dir) / train_method / f"{model}_{dataset}" / seed / "best.pth"

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
    
    
def load_dataset(dataset: str, train_method: str, data_dir: str):
    if dataset == "cifar10":
        dataset = CIFAR10(data_dir, split="test", train_method=train_method, download=True)
    elif dataset == "cifar100":
        dataset = CIFAR100(data_dir, split="test", train_method=train_method, download=True)
    return dataset


def main(
    model: Literal["densenet121", "resnet34"],
    dataset: Literal["cifar10", "cifar100"],
    train_method: Literal["ce", "logit_norm", "mixup", "openmix", "regmixup"],
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 0,
    output: str = "output.csv",
):

    # Load model and dataset    
    model = load_model(model, dataset, train_method, checkpoints_dir, seed)
    dataset = load_dataset(dataset, train_method, data_dir)

    ## TODO: Run model on dataset

    # Save Results
    outputs = None
    outputs.to_csv(output, index=True)

    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)