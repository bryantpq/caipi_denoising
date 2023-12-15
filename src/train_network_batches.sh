#!/bin/bash

config=$1
fold=$2
start_ep=1
start_batch=1
batch_size=4

for epoch in {1..50}
do
    for batch in {1..11}
    do
        if [[ $epoch -ge $start_ep ]] && [[ $batch -ge $start_batch ]]; then
            echo ""
            echo "**********************************************"
            echo "*****           Epoch ${epoch}; Batch ${batch}         *****"
            echo "**********************************************"
            echo ""
            
            cmd="python train_network_batches.py ${config} ${epoch} ${batch} --fold ${fold}"
            echo ${cmd}
            echo ""
            eval ${cmd}
        else
            echo "Skipping ${epoch} ${batch}";
        fi

        # run metrics on validation dataset after training last batch of the epoch
        if [[ $batch -eq 11 ]]; then
            cmd="python train_network_batches.py ${config} ${epoch} -1 --fold ${fold}"
            echo ""
            eval ${cmd}
        fi
    done
done
