#!/bin/bash -ex

source env.sh

mkdir -p $checkpoints_dir
for train_method in ce mixup openmix regmixup; do
    if [ ! -d $checkpoints_dir/$train_method ]; then
        wget https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/${train_method}.zip -P $checkpoints_dir
        unzip $checkpoints_dir/${train_method}.zip -d $checkpoints_dir
        mkdir -p $checkpoints_dir/${train_method}
        mv $checkpoints_dir/checkpoints/${train_method}/* $checkpoints_dir/${train_method}
        rm -rf $checkpoints_dir/${train_method}.zip
        rm -rf $checkpoints_dir/checkpoints
    fi
done

# logit_norm requires special processing
if [ ! -d $checkpoints_dir/logit_norm ]; then
    wget https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/lognorm.zip -P $checkpoints_dir
    unzip $checkpoints_dir/lognorm.zip -d $checkpoints_dir
    mkdir -p $checkpoints_dir/logit_norm
    mv $checkpoints_dir/checkpoints/lognorm/* $checkpoints_dir/logit_norm
    rm -rf $checkpoints_dir/lognorm.zip
    rm -rf $checkpoints_dir/checkpoints
    for dir in $checkpoints_dir/logit_norm/*; do
        mkdir -p $dir/1
        mv $dir/last.pt $dir/1/last.pt
    done
fi
