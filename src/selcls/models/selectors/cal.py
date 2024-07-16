
from typing import Literal
import torch
from ..calibration import TSCalibrator, DPCalibrator

class MSPCalSelector:

    def __init__(self, calibration: Literal["ts", "dp"] = "dp", n_classes: int = 10, random_state = None):
        
        if calibration == "ts":
            self.calibrator = TSCalibrator(n_classes)
        elif calibration == "dp":
            self.calibrator = DPCalibrator(n_classes)
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

