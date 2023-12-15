#!/bin/bash

experiment="compleximage_2d_full"
dims=2
model_type="cdncnn"
input_size="384 384"
path="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,outputs}"
#patching="--extract_patches --extract_step 32 32 32"
patching=""

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2023-12-12/${model_type}_ep17.h5"
"/home/quahb/caipi_denoising/models/${experiment}_fold2_2023-12-12/${model_type}_ep09.h5"
"/home/quahb/caipi_denoising/models/${experiment}_fold3_2023-12-12/${model_type}_ep03.h5"
"/home/quahb/caipi_denoising/models/${experiment}_fold4_2023-12-12/${model_type}_ep10.h5"
"/home/quahb/caipi_denoising/models/${experiment}_fold5_2023-12-12/${model_type}_ep05.h5"
)

for idx in ${!models[@]}
do
    fold=$((${idx} + 1))
    model=${models[$idx]}
    cmd="python predict.py --type_dir ${dims} ${model_type} mse ${path} ${model} ${input_size} ${patching} --fold ${fold}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
