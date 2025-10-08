#!/bin/bash

data_type='cora'
max_len=512

epochs=600
gpu=3
cls_lambda=1.0 
sparsity_lambda=1.0 
continuity_lambda=1.0

for graph_backbone in "GAT" "MLP" "GCN" "GIN" "SAGE" 
do
        log_dir=./logging/$data_type/$graph_backbone/
        mkdir -p $log_dir
        python -u main_tagr.py --max_len $max_len --embedding_dim 100 --hidden_dim 128 \
        --epochs $epochs --lr 0.0002 \
        --gpu $gpu \
        --graph_backbone $graph_backbone \
        --gnn_layers 2 \
        --data_type $data_type \
        --sparsity_percentage 0.5 \
        --cls_lambda $cls_lambda \
        --sparsity_lambda $sparsity_lambda \
        --continuity_lambda $continuity_lambda  > $log_dir/cmd.log

done



