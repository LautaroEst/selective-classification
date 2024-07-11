#!/bin/bash -ex

source env.sh

data=cifat10_vgg19_bn

mkdir -p outputs/plots/$data/
python -m selcls.scripts.plot_curves \
    --data_path $data_dir/$data \
    --output_dir outputs/plots/$data/