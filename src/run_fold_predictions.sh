#!/bin/bash

script="predict_torch.py"
experiment="compleximage_2d_full"
dims=2
model_type="cdncnn"
path="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,outputs}"

# Choice A: 2D 
input_size="--input_size 384 384"
patching=""

# Choice B: 3D
#input_size="--input_size 64 64 64"
#patching="--extract_patches --extract_step 32 32 32"

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-02-28/${model_type}_ep100.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold2_2024-02-29/${model_type}_ep100.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold3_2024-03-01/${model_type}_ep100.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold4_2024-03-02/${model_type}_ep100.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold5_2024-03-03/${model_type}_ep100.pt"
)

for idx in ${!models[@]}
do
    fold=$((${idx} + 1))
    model=${models[$idx]}
    cmd="python ${script} --type_dir --fold ${fold} ${dims} ${path} ${model} ${input_size} ${patching}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
