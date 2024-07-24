
from functools import partial
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

from .utils import read_results
from ...metrics import compute_metrics, SUPPORTED_METRICS


def process_results(row, test_lists):
    test_idx = pd.read_csv(test_lists[row["dataset"]], index_col=None, header=None).values.flatten()
    logits = pd.read_csv(row["logits_path"], index_col=0, header=0).loc[test_idx].values.astype(float)
    scores = pd.read_csv(row["scores_path"], index_col=0, header=0).loc[test_idx].values.flatten().astype(float)
    targets = pd.read_csv(row["targets_path"], index_col=0, header=0).loc[test_idx].values.flatten().astype(int)
    metrics = compute_metrics(logits, targets, scores)
    return pd.Series(metrics)


def plots(results, output_dir):
    grouped = results.groupby(["model", "dataset", "train_method"])
    for (model, dataset, train_method), group in tqdm(grouped):
        fig, ax = plt.subplots()
        for score in group["score"].unique():
            for train_list in group["train_list"].unique():
                for hparams in group["hparams"].unique():
                    group = group[
                        (group["score"] == score) &
                        (group["train_list"] == train_list) &
                        (group["hparams"] == hparams)
                    ]
                    if len(group) == 0:
                        continue
                    
                    ## TODO: plot roc

        ax.set_title(f"{model} - {dataset} - {train_method}")
        ax.grid(True)
        plt.savefig(fig, Path(output_dir) / f"{model}_{dataset}_{train_method}.png")
        fig.close()


def main(
    output_dir: str,
    eps: float = 0.0,
    temperature: float = 1.0,
    **test_lists
):
    test_lists = {k.split(".")[1]: Path(v) for k, v in test_lists.items()}
    results = read_results()
    results = results[(results["eps"] == eps) & (results["temperature"] == temperature)].reset_index(drop=True)
    results = results.loc[:,[c for c in results.columns if c not in ["eps", "temperature"]]]
    results.loc[:, SUPPORTED_METRICS] = results.progress_apply(partial(process_results, test_lists=test_lists), axis=1)
    results.to_csv(Path(output_dir) / f"raw_results.csv", index=False, header=True)
    metrics = results.groupby(["model", "dataset", "train_method", "score", "train_list", "hparams_name"]).agg({metric: ["mean", "std"] for metric in SUPPORTED_METRICS})
    metrics.to_csv(Path(output_dir) / f"metrics.csv", index=True, header=True)
    # plots(results, output_dir)



    
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)