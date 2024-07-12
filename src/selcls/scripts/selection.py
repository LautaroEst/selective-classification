


from pathlib import Path
import pickle
from typing import Literal

import torch
import pandas as pd

from ..models.selectors import MSPSelector, TSCalSelector, DPCalSelector, EntropySelector, ODINSelector, DoctorSelector, RelUSelector


def main(
    selection_method: Literal["msp", "tscal", "dpcal", "entropy", "odin", "doctor", "relu"],
    scores: str,
    train_list: str,
    predict_list: str,
    output: str,
    seed: int,
    selector_state_dict_output: str
):

    # Load scores    
    scores_path = Path(scores)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found in {scores_path}")
    scores = pd.read_csv(scores_path, index_col=0, header=0)
    logits, targets = scores.iloc[:, :-1].values, scores.iloc[:, -1].values
    
    train_idx = pd.read_csv(train_list, header=None).values.flatten()
    train_logits, train_targets = torch.from_numpy(logits[train_idx]).float(), torch.from_numpy(targets[train_idx]).long()
    train_lobprobs = torch.log_softmax(train_logits, dim=1)

    predict_idx = pd.read_csv(predict_list, header=None).values.flatten()
    predict_logits, predict_targets = torch.from_numpy(logits[predict_idx]).float(), torch.from_numpy(targets[predict_idx]).long()
    predict_lobprobs = torch.log_softmax(predict_logits, dim=1)

    # Calibrate
    n_classes = train_lobprobs.size(1)
    if selection_method == "msp":
        selector = MSPSelector(n_classes, random_state=seed)
    elif selection_method == "tscal":
        selector = TSCalSelector(n_classes, random_state=seed)
    elif selection_method == "dpcal":
        selector = DPCalSelector(n_classes, random_state=seed)
    elif selection_method == "entropy":
        selector = EntropySelector(n_classes, random_state=seed)
    elif selection_method == "odin":
        selector = ODINSelector(n_classes, random_state=seed)
    elif selection_method == "doctor":
        selector = DoctorSelector(n_classes, random_state=seed)
    elif selection_method == "relu":
        selector = RelUSelector(n_classes, random_state=seed)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    selector.fit(train_lobprobs, train_targets)
    sel_scores = selector.compute_score(predict_lobprobs)
    refined_logprobs = selector.compute_logprobs(predict_lobprobs)

    # Save
    refined_logprobs = refined_logprobs.numpy().astype(float)
    sel_scores = sel_scores.numpy().astype(float)
    predict_targets = predict_targets.numpy().astype(int)
    outputs = pd.DataFrame(refined_logprobs, columns=scores.columns[:-1], index=predict_idx)
    outputs["selection_score"] = sel_scores
    outputs["target"] = predict_targets
    outputs.to_csv(output, index=True, header=True)
    with open(selector_state_dict_output, "wb") as f:
        pickle.dump(selector.state_dict(), f)
    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)