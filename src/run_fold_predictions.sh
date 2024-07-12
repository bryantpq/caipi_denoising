#!/bin/bash

experiment="compleximage_2d_full"
out_name="outputs"

SCRIPT="predict_torch.py"
PATH="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,${out_name}}"

if [[ $experiment =~ "full" ]]; then
    dims=2
    input_size="--input_size 384 384";
    patching="";
else
    dims=3
    input_size="--input_size 64 64 64";
    patching="--extract_patches --extract_step 32 32 32";
fi

if [[ $experiment =~ "magnitude" ]]; then
    model_type="dncnn";
else
    model_type="cdncnn";
fi

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-06-28/${model_type}_ep100.pt"
#"/home/quahb/caipi_denoising/models/${experiment}_fold2_2024-07-02/${model_type}_ep100.pt"
#"/home/quahb/caipi_denoising/models/${experiment}_fold3_2024-07-02/${model_type}_ep100.pt"
#"/home/quahb/caipi_denoising/models/${experiment}_fold4_2024-07-07/${model_type}_ep100.pt"
#"/home/quahb/caipi_denoising/models/${experiment}_fold5_2024-06-28/${model_type}_ep100.pt"
)

for idx in ${!models[@]}
do
    fold=$((${idx} + 1))
    model=${models[$idx]}
    cmd="python ${SCRIPT} --type_dir --fold ${fold} ${dims} ${PATH} ${model} ${input_size} ${patching}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
