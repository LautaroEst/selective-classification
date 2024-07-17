# #!/bin/bash -ex

start=$(date +%s)

source env.sh

datasets=(cifar10 cifar100)
models=(resnet34 densenet121)
train_methods=(ce logit_norm mixup openmix regmixup)

device="cuda:0"
batch_size=128

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
                        --model $model \
                        --dataset $dataset \
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


perturbations=(0.0 0.001 0.004)
temperatures=(1.0 0.5 2.0)
scores=("msp" "entropy" "gini" "relu" "mspcal-ts" "mspcal-dp")
train_list="train"

lbd=0.5
cal_kwargs="--lr 0.1 --max_ls 100 --max_epochs 100 --tol 1e-6"

for eps in ${perturbations[@]}; do
    for temp in ${temperatures[@]}; do
        for score in ${scores[@]}; do
            if [ $score == "relu" ]; then
                kwargs="--lbd $lbd"
                kwargs_short="lbd$lbd"
            elif [ $score == "mspcal-ts" ] || [ $score == "mspcal-dp" ]; then
                kwargs=$cal_kwargs
                kwargs_short="default"
            else
                kwargs=""
                kwargs_short="none"
            fi
            for dataset in ${datasets[@]}; do
                for model in ${models[@]}; do
                    for train_method in ${train_methods[@]}; do
                        for dir in $checkpoints_dir/$train_method/${model}_${dataset}/*; do
                            seed=$(basename $dir)
                            posteriors_dir="outputs/img_classification/$dataset/$model/$train_method/seed=$seed"
                            output_dir="$posteriors_dir/eps=$eps/temp=$temp/score=$score/train_list=$train_list/hparams=$kwargs_short"
                            if [ ! -f $output_dir/logits.csv ]; then
                                mkdir -p $output_dir
                                python -m selcls.scripts.img_classification.selection_scores \
                                    --model $model \
                                    --dataset $dataset \
                                    --train_method $train_method \
                                    --input_perturbation $eps \
                                    --temperature $temp \
                                    --score $score \
                                    --logits "$posteriors_dir/logits.csv" \
                                    --targets "$posteriors_dir/targets.csv" \
                                    --train_list $lists_dir/$dataset/$train_list \
                                    --data_dir $data_dir/${dataset}_data \
                                    --checkpoints_dir $checkpoints_dir \
                                    --seed $seed \
                                    --output_dir $output_dir \
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

end=$(date +%s)
echo "Time taken: $(($((end-start))/3600))h$(($(($((end-start))%3600))/60))m$(($((end-start))%60))s"