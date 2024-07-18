from pathlib import Path

import torch
from torch import nn

from .desenet import DenseNet121Small
from .resnet import ResNet34

class OpenMixModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)[:, :-1]
    


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
    if train_method == "openmix":
        model = OpenMixModel(model)

    return model