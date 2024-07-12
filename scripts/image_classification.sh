#!/bin/bash -ex

source env.sh

datasets=(cifar10 cifar100)
models=(resnet34 densenet121)
train_methods=(ce logit_norm mixup openmix regmixup)

device=cuda:0
batch_size=128

for dataset in ${datasets[@]}; do
    for model in ${models[@]}; do
        for train_method in ${train_methods[@]}; do
            for dir in $checkpoints_dir/$train_method/${model}_${dataset}/*; do
                seed=$(basename $dir)
                output="outputs/$dataset/$model/$train_method/seed=$seed/outputs.csv" 
                if [ ! -f $output ]; then
                    mkdir -p $(dirname $output)
                    python -m selcls.scripts.run_img_classification \
                        --dataset $dataset \
                        --model $model \
                        --train_method $train_method \
                        --data_dir $data_dir/${dataset}_data \
                        --checkpoints_dir $checkpoints_dir \
                        --seed $seed \
                        --output $output \
                        --device $device \
                        --batch_size $batch_size
                fi
            done
        done
    done
done