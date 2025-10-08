# TaGR

This repository contains code for the paper "Rationalizing Text-attributed Graph Representation Learning". 
We release some key code and hyperparameters in experiments for anonymous review. We will release all the code used in experiments upon acceptance.
This work has been submitted to The Web Conference 2026 for possible publication.

## Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.8.

We suggest you to create a virtual environment with: conda create -n TaGR python=3.8.20

Then activate the environment with: conda activate TaGR 

Install packages: pip install -r requirements.txt

## Running example
### Cora Dataset
~~~
data_type=cora
graph_backbone=GAT
log_dir=./logging/$data_type/$graph_backbone/
mkdir -p $log_dir
python -u main_tagr.py --max_len 512 --embedding_dim 100 --hidden_dim 128 \
        --epochs 600 --lr 0.0002 \
        --gpu 3 \
        --graph_backbone $graph_backbone \
        --gnn_layers 2 \
        --data_type $data_type \
        --sparsity_percentage 0.5 \
        --cls_lambda 1.0 \
        --sparsity_lambda 1.0 \
        --continuity_lambda 1.0  > $log_dir/cmd.log	
~~~

**_Notes_**: "--sparsity_percentage 0." means "$s=0.5$" in Sect. 4.2 (But the actual sparsity is different from $s$. When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set.). "--sparsity_lambda 1.0 --continuity_lambda 1.0 " means $\lambda_1=1.0, \lambda_2=1.0$. " 
"--epochs 600" means we run 600 epochs and take the results when the "dev_acc" is best.

## Result  
You will obtain the result in the Cora folder located under the $log_dir directory. Then, you need to locate the result within the corresponding log file.

For Cora dataset, you may get a result like: 
~~~
vali_s: mean=0.5443, std=0.2195
vali_acc: mean=0.8641, std=0.0140
vali_rat_p: mean=0.5250, std=0.1090
vali_rat_r: mean=0.6676, std=0.3852
vali_rat_f1: mean=0.5673, std=0.2436

test_s: mean=0.5786, std=0.2219
test_acc: mean=0.8836, std=0.0268
test_rat_p: mean=0.5627, std=0.0048
test_rat_r: mean=0.8005, std=0.0230
test_rat_f1: mean=0.6607, std=0.0056

~~~

The line "test_acc: mean=0.8836, std=0.0268" and "test_rat_f1: mean=0.6607, std=0.0056"  indicate that the classification Accuracy and  rationale F1 score (%) are 88.36 and 66.07, respectively.


## Dependencies
- torch==1.12.1
- matplotlib==3.7.5
- numpy==1.26.3
- pandas==2.0.3



