#!/bin/bash

experiment="sr_magnitude_patch"
out_name="outputs_l2_res_1e-05"

params="--residual"; # options: [ --residual, --rescale ]

model_type="my_dcsrn";
batch_size=12;

models=(
"/home/quahb/caipi_denoising/models/${experiment}_fold1_2024-09-23/${model_type}_ep98.pt"
)

MYSCRIPT="predict_torch.py"
MYPATH="/home/quahb/caipi_denoising/data/datasets/accelerated/${experiment}/{inputs,${out_name}}"

if [[ $experiment =~ "full" ]]; then
    dims=2
    input_size="--input_size 384 384";
    patching="";
    batching="--batch_size ${batch_size}"
else
    dims=3
    input_size="--input_size 64 64 64";
    patching="--extract_patches --extract_step 32 32 32";
    batching="--batch_size ${batch_size}"
fi

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
