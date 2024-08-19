#!/bin/bash

#experiment="magnitude_3d_patches64"
experiment="compleximage_2d_full"
out_name="outputs_l2"

params="--rescale --residual"; # options: [ --residual, --rescale ]

if [[ $experiment =~ "magnitude" ]]; then model_type="dncnn";
else model_type="cdncnn";
fi

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-08-14/${model_type}_ep56.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold2_2024-08-15/${model_type}_ep96.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold3_2024-08-16/${model_type}_ep80.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold4_2024-08-17/${model_type}_ep71.pt"
"/home/quahb/caipi_denoising/models/${experiment}_fold5_2024-08-18/${model_type}_ep59.pt"
)

MYSCRIPT="predict_torch.py"
MYPATH="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,${out_name}}"

if [[ $experiment =~ "full" ]]; then
    dims=2
    input_size="--input_size 384 384";
    patching="";
else
    dims=3
    input_size="--input_size 64 64 64";
    patching="--extract_patches --extract_step 32 32 32";
fi

for idx in ${!models[@]}
do
    model=${models[$idx]}
    fold=${model#*fold}
    fold=`cut -d "_" -f1 <<< "$fold"`
    cmd="python ${MYSCRIPT} --type_dir ${params} --fold ${fold} ${dims} ${MYPATH} ${model} ${input_size} ${patching}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
