# Experiments on Selective Classification

## Installation

```bash
conda create --name selcls python=3.11.9
conda activate selcls
pip install -r requirements.txt
pip install -e .
```

## Usage

Before runing any script, make sure to setup all the environment variables in the `env.sh` file.

```bash

...
```

In addition, make executable all the scripts in the `scripts/` directory.

```bash
chmod -R +x scripts/
```

Then, you can run the scripts to reproduce the results. Make sure you are working under the `refind` conda environment.
```
conda activate refind
./scripts/run...sh
```
