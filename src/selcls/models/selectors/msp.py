

import torch

class MSPSelector:

    def __init__(self, n_classes, random_state = None):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, train_lobprobs, train_targets):
        self.train_logprobs = train_lobprobs
        self.train_targets = train_targets

    def compute_score(self, predict_lobprobs):
        return torch.softmax(predict_lobprobs, dim=1).max(dim=1).values
    
    def compute_logprobs(self, predict_lobprobs):
        return predict_lobprobs
