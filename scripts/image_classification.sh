#!/bin/bash -ex

source env.sh

datasets=(cifar10 cifar100)
models=(resnet34 densenet121)
train_methods=(ce logit_norm mixup openmix regmixup)

for dataset in ${datasets[@]}; do
    for model in ${models[@]}; do
        for train_method in ${train_methods[@]}; do
            for dir in $checkpoints_dir/$train_method/${model}_${dataset}/*; do
                seed=$(basename $dir)
                output_dir="outputs/img_classification/$dataset/$model/$train_method/seed=$seed"
                logits_output=$output_dir/logits.csv 
                targets_output=$output_dir/targets.csv
                if [ ! -f $logits_output ]; then
                    mkdir -p $output_dir
                    python -m selcls.scripts.img_classification.posteriors \
                        --dataset $dataset \
                        --model $model \
                        --train_method $train_method \
                        --data_dir $data_dir/${dataset}_data \
                        --checkpoints_dir $checkpoints_dir \
                        --seed $seed \
                        --logits_output $logits_output \
                        --targets_output $targets_output \
                        --device $device \
                        --batch_size $batch_size
                fi
            done
        done
    done
done


perturbations=(0.001 0.004)
temperatures=(0.5 2.0)
scores=("msp" "entropy" "gini" "relu" "mspcal-ts" "mspcal-dp")
lbd=0.5

for eps in ${perturbations[@]}; do
    for temp in ${temperatures[@]}; do
        for score in ${scores[@]}; do
            if [ $score == "relu" ]; then
                kwargs="--lbd $lbd"
            else
                kwargs=""
            fi
            kwargs_name=$(echo $kwargs | sed 's/ --/_/g' | sed 's/ /=/g' | sed 's/--/_/g')
            for dataset in ${datasets[@]}; do
                for model in ${models[@]}; do
                    for train_method in ${train_methods[@]}; do
                        for dir in $checkpoints_dir/$train_method/${model}_${dataset}/*; do
                            seed=$(basename $dir)
                            output_dir="outputs/img_classification/$dataset/$model/$train_method/seed=$seed/eps=$eps/temp=$temp/score=$score"
                            logits_output=$output_dir/logits$kwargs_name.csv 
                            targets_output=$output_dir/targets$kwargs_name.csv
                            if [ ! -f $logits_output ]; then
                                mkdir -p $output_dir
                                python -m selcls.scripts.img_classification.selection_scores \
                                    --model $model \
                                    --dataset $dataset \
                                    --train_method $train_method \
                                    --input_perturbation $eps \
                                    --temperature $temp \
                                    --score $score \
                                    --logits "outputs/img_classification/$dataset/$model/$train_method/seed=$seed/logits.csv" \
                                    --targets "outputs/img_classification/$dataset/$model/$train_method/seed=$seed/targets.csv" \
                                    --train_list $lists_dir/train \
                                    --data_dir $data_dir/${dataset}_data \
                                    --checkpoints_dir $checkpoints_dir \
                                    --seed $seed \
                                    --logits_output $logits_output \
                                    --targets_output $targets_output \
                                    --device $device \
                                    --batch_size $batch_size \
                                    $kwargs
                            fi
                        done
                    done
                done
            done
        done
    done
done
