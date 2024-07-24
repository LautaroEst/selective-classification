#!/bin/bash -ex

source env.sh

model_metrics=accuracy,norm_cross_entropy
sel_metrics=fpr95tpr,
output_dir=outputs/img_classification/results
eval_set=test

mkdir -p $output_dir
python -m selcls.scripts.img_classification.base_results \
    --model_metrics $model_metrics \
    --sel_metrics $sel_metrics \
    --output_dir $output_dir \
    --test_list.cifar10 $lists_dir/cifar10/$eval_set \
    --test_list.cifar100 $lists_dir/cifar100/$eval_set