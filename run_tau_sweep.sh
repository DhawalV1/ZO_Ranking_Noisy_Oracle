#!/bin/bash

# TAUS=(
# 0.01
# 0.05
# 0.1
# 0.15
# 0.2
# 0.3
# 0.5
# )

TAUS=(0.0 0.01 0.05 0.1)
for TAU in "${TAUS[@]}"
do

echo "===================================="
echo "Running tau = $TAU"
echo "===================================="

python train_kt.py \
    --tau $TAU \
    --result_dir results_tau_ls_${TAU}

done

# ls denotes implementation with line search
