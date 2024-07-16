

import torch
from ...scores import entropy

class EntropySelector:

    def __init__(self, n_classes, temperature, random_state = None):
        self.n_classes = n_classes
        self.temperature = temperature
        self.random_state = random_state

    def fit(self, train_logits, train_targets):
        self.train_logits = train_logits
        self.train_targets = train_targets

    def compute_score(self, predict_logits):
        return entropy(predict_logits, self.temperature)
    
    def compute_logprobs(self, predict_logits):
        return torch.log_softmax(predict_logits / self.temperature, dim=1)