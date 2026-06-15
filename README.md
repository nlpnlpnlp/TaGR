# 🧩 Towards Trustworthy Graph-Text Learning: Cooperative Rationalization with Causal Self-Distillation

[![Paper Status](https://img.shields.io/badge/Status-TMM--2026--Submission-orange)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.8.0-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)](https://developer.nvidia.com/cuda-toolkit)


This repository contains code for the paper "Towards Trustworthy Graph-Text Learning: Cooperative Rationalization with Causal Self-Distillation". 
We release some key code and hyperparameters in experiments for anonymous review. We will release all the code used in experiments upon acceptance.
This work has been submitted to the IEEE for possible publication.


## 📘 Overview
Graph structures can model interconnected entities in many social services, supporting a wide range of multimedia applications. However, nodes and edges in real-world graphs are often associated with intrinsic textual attributes, which not only connect nodes through the graph topology to reflect linking relations but also associate each node’s feature space with human-interpretable textual semantics. This raises a new research question: Can we construct a unified graph learning model for such graphs that jointly captures both structural and textual modalities, while explicitly aligning learned node representations with human-interpretable textual semantics to enable transparent and trustworthy graph-text learning? To address this challenge, we propose a novel explainable graph-text learning method, called TaGR (**T**ext-**a**ttributed **G**raph **R**ationalizer). 
Specifically, we introduce a cooperative game–based rationalization framework, which consists of a rationale generator that highlights graph rationales and a predictor responsible for message passing. Notably, they are jointly optimized in a cooperative manner. Furthermore, to improve the quality of the rationales, we propose a self-distillation strategy with causal decomposition to align the rationale space with the prediction space. Finally, Extensive experiments on four benchmark datasets demonstrate that the proposed method not only improves predictive accuracy but also significantly enhances the quality of self-explanations. This improvement is especially valuable for multimedia applications, as mining intrinsic explanations plays a crucial role in effectively understanding the model’s internal information flow.


## 🏗️ Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.8.

We suggest you to create a virtual environment with: conda create -n TaGR python=3.8.20

Then activate the environment with: conda activate TaGR 

Install packages: pip install -r requirements.txt

## 🚀 Running example
### Cora Dataset
For example, on the Cora dataset with GAT as the backbone model, you need to run the following script:
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
📝 **_Notes_**: "--sparsity_percentage 0.5" means "$s$=0.5" in Sect. 4.2 (But the actual sparsity is different from $s$. When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set). "--sparsity_lambda 1.0 --continuity_lambda 1.0 " means $\lambda_1=1.0, \lambda_2=1.0$. "--epochs 600" means we run 600 epochs and take the results.

## 📊 Result  
You will obtain the result in the Cora folder located under the $log_dir directory. Then, you need to locate the result within the corresponding log file.

For Cora dataset, you may get a result like: 
~~~
vali_s: mean=0.5746, std=0.0164
vali_acc: mean=0.8864, std=0.0115
vali_rat_p: mean=0.5517, std=0.0067
vali_rat_r: mean=0.7994, std=0.0196
vali_rat_f1: mean=0.6527, std=0.0065

test_s: mean=0.5786, std=0.0200
test_acc: mean=0.8836, std=0.0268
test_rat_p: mean=0.5627, std=0.0048
test_rat_r: mean=0.8005, std=0.0230
test_rat_f1: mean=0.6607, std=0.0056
~~~
The line "test_acc: mean=0.8836, std=0.0268" and "test_rat_f1: mean=0.6607, std=0.0056"  indicate that the classification Accuracy and  rationale F1 score are 0.8836 and 0.6607, respectively. Therefore, you can obtain the classification accuracy and rationale F1 score on a percentage scale: 88.36% and 66.07%, respectively.

If you want to conduct experiments using TAPE as the baseline, you first need to obtain the TAPE-pretrained embeddings (located in the data folder).
Then, you should run the following script:
~~~
python enhanced.py --data_type cora
~~~

## 🔗 Dependencies
- torch==1.12.1
- matplotlib==3.7.5
- numpy==1.26.3
- pandas==2.0.3


