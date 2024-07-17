
import torch
from .base import BaseSelector

class EntropySelector(BaseSelector):

    def __init__(self, n_classes, random_state = None):
        super().__init__()
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, train_logits, train_targets):
        self.train_logits = train_logits
        self.train_targets = train_targets

    def compute_score(self, predict_logits):
        log_probs = torch.log_softmax(predict_logits, dim=1)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)
    
    def compute_logprobs(self, predict_logits):
        return torch.log_softmax(predict_logits, dim=1)