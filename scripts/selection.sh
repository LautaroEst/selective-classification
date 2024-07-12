#!/bin/bash -ex

python -m selcls.scripts.selection \
    --selection_method "dpcal" \
    --scores "outputs/cifar10/resnet34/mixup/seed=2/outputs.csv" \
    --train_list "lists/cifar10/train" \
    --predict_list "lists/cifar10/test" \
    --output "outputs/cifar10/resnet34/mixup/seed=2/dpcal/train=train_predict=test_selection.csv" \
    --seed 2 \
    --selector_state_dict_output "outputs/cifar10/resnet34/mixup/seed=2/dpcal/state_dict.pkl" \