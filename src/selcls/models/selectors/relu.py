
import torch
from torch.utils.data import DataLoader, TensorDataset

class RelUSelector:

    def __init__(self, n_classes, lbd = 0.5, temperature = 1.0, batch_size = 128, random_state = None):
        self.n_classes = n_classes
        self.lbd = lbd
        self.temperature = temperature
        self.batch_size = batch_size
        self.random_state = random_state
        self.params = None

    def fit(self, train_logits, train_targets):
        train_pred = train_logits.argmax(dim=1)
        train_labels = (train_targets != train_pred).int()

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

        self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
        self.params = torch.tril(self.params, diagonal=-1)
        self.params = self.params + self.params.T
        self.params = torch.relu(self.params)
        if torch.all(self.params <= 0):
            # default to gini
            self.params = torch.ones(self.params.size()).to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
        self.params = self.params / self.params.norm()

    def compute_score(self, predict_logits):
        if self.params is None:
            raise ValueError("Selector not fitted")

        probs = torch.softmax(predict_logits / self.temperature, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)
    
    def compute_logprobs(self, predict_logits):
        return torch.log_softmax(predict_logits / self.temperature, dim=1)