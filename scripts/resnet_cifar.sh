#!/bin/bash -ex

python -m selcls.scripts.train_and_run_resnet --dataset cifar10 --model resnet18 --batch-size 4