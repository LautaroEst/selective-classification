
from torch import nn

class BaseSelector(nn.Module):

    def fit(self, train_logits, train_targets):
        raise NotImplementedError
    
    def compute_score(self, predict_logits):
        raise NotImplementedError
    
    def compute_logprobs(self, predict_logits):
        raise NotImplementedError
    
    @property
    def hparams(self):
        return {}