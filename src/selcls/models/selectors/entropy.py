

import torch

class EntropySelector:

    def __init__(self, n_classes, random_state = None):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, train_lobprobs, train_targets):
        self.train_logprobs = train_lobprobs
        self.train_targets = train_targets

    def compute_score(self, predict_lobprobs):
        probs = torch.softmax(predict_lobprobs, dim=1)
        return -torch.sum(probs * predict_lobprobs, dim=1)
    
    def compute_logprobs(self, predict_lobprobs):
        return predict_lobprobs