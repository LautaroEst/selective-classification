
from collections import OrderedDict
from typing import Literal
import torch
from ..calibration import TSCalibrator, DPCalibrator

class MSPCalSelector:

    def __init__(self, calibration: Literal["ts", "dp"] = "dp", n_classes: int = 10, device: str = "cuda:0", random_state = None, **kwargs):
        
        self.device = torch.device(device)
        if calibration == "ts":
            self.calibrator = TSCalibrator(n_classes, **kwargs).to(self.device)
        elif calibration == "dp":
            self.calibrator = DPCalibrator(n_classes, **kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")

        self.n_classes = n_classes
        self.random_state = random_state        

    def fit(self, train_logits, train_targets):
        train_logprobs = torch.log_softmax(train_logits, dim=1)
        self.calibrator.fit(train_logprobs, train_targets)

    def compute_score(self, predict_logits):
        predict_logprobs = torch.log_softmax(predict_logits, dim=1)
        cal_logprobs = self.calibrator.calibrate(predict_logprobs)
        return torch.softmax(cal_logprobs, dim=1).max(dim=1).values
    
    def compute_logprobs(self, predict_logits):
        predict_logprobs = torch.log_softmax(predict_logits, dim=1)
        return self.calibrator.calibrate(predict_logprobs)

    @property
    def hparams(self):
        return {
            "lr": self.calibrator.lr,
            "max_ls": self.calibrator.max_ls,
            "max_epochs": self.calibrator.max_epochs,
            "tol": self.calibrator.tol,
        }
    
    def state_dict(self):
        return OrderedDict([(name, param.detach().cpu()) for name, param in self.calibrator.state_dict().items()])
    
    def load_state_dict(self, state_dict):
        state_dict = OrderedDict([(name, param.to(self.device)) for name, param in state_dict.items()])
        self.calibrator.load_state_dict(state_dict)