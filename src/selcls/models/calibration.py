
import torch
from torch import nn


class BaseCalibrator(nn.Module):
    
    def calibrate(self, logprobs):
        self.eval()
        with torch.no_grad():
            cal_logprobs = self(logprobs)
        return cal_logprobs
    
    def fit(self, logprobs, labels):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2, max_iter=100)

        def closure():
            optimizer.zero_grad()
            cal_logprobs = self(logprobs)
            loss = criterion(cal_logprobs, labels)
            loss.backward()
            return loss
        
        last_loss = float("inf")
        for epoch in range(100):
            loss = optimizer.step(closure)
            if abs(last_loss - loss) < 1e-6:
                break

        return self
    

class DPCalibrator(BaseCalibrator):
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        return self.alpha * x + self.beta


class TSCalibrator(BaseCalibrator):
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x / self.temp