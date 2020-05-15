#!/usr/bin/env bash
current_dir=$(cd `dirname $0`; pwd)
echo ${current_dir}

userHome=`whoami`
cd ${current_dir}/


source /Users/mac/anaconda3/bin/activate pytorch
# source /opt/conda/bin/activate pytorch
python3 ./bin/train.py