#!/bin/bash

# Hyperparameter values
LEARNING_RATES=(0.1 0.01 0.001 0.0001)
WEIGHT_DECAYS=(0.1 0.01 0.001 0.0001)
BATCH_SIZES=(1 2 4 8 16)
EPOCHS=(10 20 40 60 80 100)
PATCH_SIZES=(4 8 16 32 64)
NUM_HEADS=(4 8 16)
TRANSFORMER_LAYERS=(4 8 12 24)

for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for ep in "${EPOCHS[@]}"; do
                for ps in "${PATCH_SIZES[@]}"; do
                    for nh in "${NUM_HEADS[@]}"; do
                        for tl in "${TRANSFORMER_LAYERS[@]}"; do
                            echo "Training started for LR=$lr, WD=$wd, BS=$bs, EP=$ep, PS=$ps, NH=$nh, TL=$tl"
                            python EarthRemoteSensingRapidResponse/ERSRR_MODEL.py --tune True --lr "$lr" --wd "$wd" --bs "$bs" --ep "$ep" --ps "$ps" --nh "$nh" --tl "$tl"

                            echo "Finished"
                        done
                    done
                done
            done
        done
    done
done

#awk '{ if ($9 < min) min = $9 } END { print min }' EarthRemoteSensingRapidResponse/ParamTest/tuningOutput.txt