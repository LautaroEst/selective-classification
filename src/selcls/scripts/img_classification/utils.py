


from pathlib import Path
import pandas as pd


def read_results():
    results = []
    outputs_path = Path(f"outputs/img_classification")
    for dataset_path in outputs_path.iterdir():
        if dataset_path.name == "results":
            continue
        for model_path in dataset_path.iterdir():
            for train_method_path in model_path.iterdir():
                for seed_path in train_method_path.iterdir():
                    seed = int(seed_path.name.split("=")[1])
                    for eps_path in seed_path.iterdir():
                        if not eps_path.is_dir():
                            continue
                        eps = float(eps_path.name.split("=")[1])
                        for temperature_path in eps_path.iterdir():
                            temperature = float(temperature_path.name.split("=")[1])
                            for score_path in temperature_path.iterdir():
                                score = score_path.name.split("=")[1]
                                for train_list_path in score_path.iterdir():
                                    for hparams_path in train_list_path.iterdir():
                                        logits_path = hparams_path / "logits.csv"
                                        targets_path = hparams_path / "targets.csv"
                                        scores_path = hparams_path / "scores.csv"
                                        results.append({
                                            "dataset": dataset_path.name,
                                            "model": model_path.name,
                                            "train_method": train_method_path.name,
                                            "score": score,
                                            "train_list": train_list_path.name,
                                            "hparams_name": hparams_path.name.split("=")[1],
                                            "eps": eps,
                                            "temperature": temperature,
                                            "seed": seed,
                                            "hparams_path": str(hparams_path / "hparams.yaml"),
                                            "logits_path": str(logits_path),
                                            "targets_path": str(targets_path),
                                            "scores_path": str(scores_path),
                                        })
    return pd.DataFrame(results)