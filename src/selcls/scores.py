
import torch

def msp(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return -max_probs


def entropy(logits: torch.Tensor):
    log_probs = torch.log_softmax(logits, dim=1)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)


def gini(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    g = torch.sum(probs ** 2, dim=1)
    return (1 - g) / g