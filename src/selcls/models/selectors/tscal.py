
import torch
from ..calibration import TSCalibrator

class TSCalSelector:

    def __init__(self, n_classes):
        self.calibrator = TSCalibrator(n_classes)

    def fit(self, train_lobprobs, train_targets):
        self.calibrator.fit(train_lobprobs, train_targets)

    def compute_score(self, predict_lobprobs):
        cal_logprobs = self.calibrator.calibrate(predict_lobprobs)
        return torch.softmax(cal_logprobs, dim=1).max(dim=1).values

