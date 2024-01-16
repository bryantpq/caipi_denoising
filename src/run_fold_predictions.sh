#!/bin/bash

script="predict_torch.py"
experiment="magnitude_2d_full"
dims=2
model_type="dncnn"
input_size="384 384"
path="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,torch_f1}"
#patching="--extract_patches --extract_step 32 32 32"
patching=""

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-01-09/${model_type}_ep20.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold2_2024-01-09/${model_type}_ep16.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold3_2024-01-09/${model_type}_ep16.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold4_2024-01-09/${model_type}_ep9.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold5_2024-01-10/${model_type}_ep5.pt"
)

for idx in ${!models[@]}
do
    fold=$((${idx} + 1))
    model=${models[$idx]}
    cmd="python ${script} --type_dir --fold ${fold} ${dims} ${model_type} mse ${path} ${model} ${input_size} ${patching}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
