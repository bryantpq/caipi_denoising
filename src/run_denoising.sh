#!/bin/bash

experiment="magnitude_reg"
#experiment="magnitude_2d_full"
#experiment="compleximage_2d_full"
#experiment="magnitude_3d_patches64"
#experiment="compleximage_3d_patches64"

out_name="magnitude_2d_full"

params="--residual"; # options: [ --residual, --rescale ]

if [[ $experiment =~ "magnitude" || $out_name =~ "magnitude" ]]; then 
    echo "Using dncnn model"
    model_type="dncnn";
else 
    echo "Using cdncnn model"
    model_type="cdncnn";
fi

if [[ $experiment =~ "full" || $out_name =~ "full" ]]; then 
    echo "Processing data as 2D slices"
    dims=2
    batch_size=12;
    input_size="--input_size 384 384";
    patching="";
    batching="--batch_size ${batch_size}"
else 
    echo "Processing data as 3D patches"
    dims=3
    batch_size=16;
    input_size="--input_size 64 64 64";
    patching="--extract_patches --extract_step 32 32 32";
    batching="--batch_size ${batch_size}"
fi

models=(
#"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-09-19/${model_type}_ep75.pt"
#"/home/quahb/caipi_denoising/models/magnitude_3d_patches64_fold1_2024-08-22/dncnn_ep100.pt"
#"/home/quahb/caipi_denoising/models/magnitude_3d_patches64_fold2_2024-08-26/dncnn_ep100.pt"
#"/home/quahb/caipi_denoising/models/magnitude_3d_patches64_fold3_2024-08-28/dncnn_ep100.pt"
#"/home/quahb/caipi_denoising/models/magnitude_3d_patches64_fold4_2024-08-22/dncnn_ep90.pt "
#"/home/quahb/caipi_denoising/models/magnitude_3d_patches64_fold5_2024-08-26/dncnn_ep35.pt "
"/home/quahb/caipi_denoising/models/magnitude_2d_full_fold1_2024-09-17/dncnn_ep100.pt"
"/home/quahb/caipi_denoising/models/magnitude_2d_full_fold2_2024-09-17/dncnn_ep100.pt"
"/home/quahb/caipi_denoising/models/magnitude_2d_full_fold3_2024-09-18/dncnn_ep100.pt"
"/home/quahb/caipi_denoising/models/magnitude_2d_full_fold4_2024-09-18/dncnn_ep100.pt"
"/home/quahb/caipi_denoising/models/magnitude_2d_full_fold5_2024-09-18/dncnn_ep100.pt"
)

MYSCRIPT="predict_torch.py"
MYPATH="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,${out_name}}"

for idx in ${!models[@]}
do
    model=${models[$idx]}
    fold=${model#*fold}
    fold=`cut -d "_" -f1 <<< "$fold"`
    cmd="python ${MYSCRIPT} --type_dir ${params} --fold ${fold} ${dims} ${MYPATH} ${model} ${input_size} ${patching} ${batching}"
    echo ${cmd}
    echo ""
    eval ${cmd}
done
