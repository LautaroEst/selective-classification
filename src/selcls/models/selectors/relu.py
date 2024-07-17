
import torch
from torch import nn
from .base import BaseSelector

class RelUSelector(BaseSelector):

    def __init__(self, n_classes, lbd = 0.5, random_state = None):
        super().__init__()
        self.n_classes = n_classes
        self.lbd = lbd
        self.random_state = random_state
        self.params = nn.Parameter(torch.eye(n_classes))

    def fit(self, train_logits, train_targets):
        train_pred = train_logits.argmax(dim=1)
        train_labels = (train_targets != train_pred).int()

        train_probs = torch.softmax(train_logits, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

        params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.params.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.params.device)
        params = torch.tril(params, diagonal=-1)
        params = params + params.T
        params = torch.relu(params)
        if torch.all(params <= 0):
            # default to gini
            params = torch.ones(params.size()).to(self.params.device)
            params = torch.tril(params, diagonal=-1)
            params = params + params.T
        params = params / params.norm()
        self.params.data = params

    def compute_score(self, predict_logits):
        if self.params is None:
            raise ValueError("Selector not fitted")

        probs = torch.softmax(predict_logits, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)
    
    def compute_logprobs(self, predict_logits):
        return torch.log_softmax(predict_logits, dim=1)
    
    @property
    def hparams(self):
        return {
            "lbd": self.lbd,
        }
    
