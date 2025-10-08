


# 0.2 0.4 0.6
# "MLP" "GCN" "GIN" "SAGE" "RevGAT"
#2 3 1 4 5
for gnn_layers in  2 1 3 4 5 6 7 8 9 10
do

for sparsity_percentage in 0.6 0.2 0.4 
do
    for graph_backbone in "GCN" "MLP"  "GIN" "SAGE" "RevGAT"
    do
        data_type='cora'
        max_len=512

        epochs=600
        gpu=3

        # gnn_layers=3
        cls_lambda=100.0 
        sparsity_lambda=100.0 
        continuity_lambda=1.0
        
        log_dir=./log_res/$gnn_layers/$data_type/'spa'$sparsity_percentage/$graph_backbone/
        mkdir -p $log_dir
        python -u norm_tagr_rnp_my.py --max_len $max_len --embedding_dim 100 --hidden_dim 128 \
        --epochs $epochs --lr 0.0002 \
        --gpu $gpu \
        --graph_backbone $graph_backbone \
        --gnn_layers $gnn_layers \
        --data_type $data_type \
        --sparsity_percentage $sparsity_percentage \
        --cls_lambda $cls_lambda \
        --sparsity_lambda $sparsity_lambda \
        --continuity_lambda $continuity_lambda  > $log_dir/cmd.log


    done
done

done

