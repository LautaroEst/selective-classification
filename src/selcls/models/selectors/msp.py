

import torch

class MSPSelector:

    def __init__(self, n_classes):
        pass

    def fit(self, train_lobprobs, train_targets):
        pass

    def compute_score(self, predict_lobprobs):
        return torch.softmax(predict_lobprobs, dim=1).max(dim=1).values
