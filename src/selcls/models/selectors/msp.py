

from collections import OrderedDict
import torch

class MSPSelector:

    def __init__(self, n_classes, random_state = None):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, train_logits, train_targets):
        self.train_logits = train_logits
        self.train_targets = train_targets

    def compute_score(self, predict_logits):
        probs = torch.softmax(predict_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        return -max_probs
    
    def compute_logprobs(self, predict_logits):
        return torch.log_softmax(predict_logits, dim=1)
    
    @property
    def hparams(self):
        return {}

    def state_dict(self):
        return OrderedDict([])
    
    def load_state_dict(self, state_dict):
        pass