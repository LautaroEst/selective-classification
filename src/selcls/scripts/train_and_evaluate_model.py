

from typing import Literal
from ..model import resnet18
from ..dataset.cifar import get_loader as get_cifar_loader
from ..metrics import calc_metrics

from torch import nn


def main(
    model: Literal["resnet18"],
    dataset: Literal["cifar10", "cifar100"] = 'cifar10',
    batch_size: int = 128,
):
    
    # Load the dataset
    train_loader, valid_loader, test_loader, \
    test_onehot, test_label, num_classes = get_cifar_loader(dataset, batch_size)

    # Load the model
    model_dict = {
        "num_classes": num_classes
    }
    if model == 'resnet18':
        model = resnet18.ResNet18(**model_dict).cuda()
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    
    cls_criterion = nn.CrossEntropyLoss().cuda()

    # TODO: Train the model
    # ...

    # Print results
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier = calc_metrics(
        test_loader, test_label, test_onehot, model, cls_criterion, classnumber=num_classes
    )

if __name__ == '__main__':
    from fire import Fire
    Fire(main)