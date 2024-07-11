from torchvision import transforms
from torchvision.datasets import CIFAR10 as _CIFAR10, CIFAR100 as _CIFAR100


def get_transforms(statistics, split="train"):
    if split == "test":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize(*statistics),
            ]
        )
    elif split == "train":
        return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*statistics),
        ]
    )
    raise ValueError(f"split must be 'train' or 'test', got {split}")




class CIFAR10(_CIFAR10):

    statistics = {
        "ce": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "logit_norm": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "mixup": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "openmix": ((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        "regmixup": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    }

    def __init__(self, root: str, split = "train", method = "ce", download: bool = False):
        if split == "train":
            train = True
        elif split == "test":
            train = False
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        transform = get_transforms(self.statistics[method], split)
        super().__init__(root, train=train, transform=transform, target_transform=None, download=download)


class CIFAR100(_CIFAR100):

    statistics = {
        "ce": ((0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588)),
        "logit_norm": ((0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588)),
        "mixup": ((0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588)),
        "openmix": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        "regmixup": ((0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588)),
    }

    def __init__(self, root: str, split = "train", method = "ce", download: bool = False):
        if split == "train":
            train = True
        elif split == "test":
            train = False
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        transform = get_transforms(self.statistics[method], split)
        super().__init__(root, train=train, transform=transform, target_transform=None, download=download)