import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from model import StudentTagRnpModel2,TeacherEncoder
from reader_cora import get_raw_text_cora
from reader_pubmed import get_raw_text_pubmed,get_raw_text_pubmed_subgraph_aligned
from reader_products import get_raw_text_products ,get_raw_text_products_subgraph_aligned
from reader_arxiv_2023 import get_raw_text_arxiv_2023,get_raw_text_arxiv_2023_subgraph_aligned
from metric import compute_micro_stats
import torch.nn as nn
import sys
import re
import pandas as pd
from tqdm import tqdm

#########################
# GloVe Embedding Loader
#########################

def get_glove_embedding(glove_embedding_path):
    with open(glove_embedding_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        embedding = []
        word2idx = {}
        idx2word = {}
        for indx, line in enumerate(lines):
            word, emb = line.split()[0], line.split()[1:]
            vector = [float(x) for x in emb]
            if indx == 0:
                embedding.append(np.zeros(len(vector)))
            embedding.append(vector)
            word2idx[word] = indx + 1
            idx2word[indx + 1] = word
        embedding = np.array(embedding, dtype=np.float32)
        return embedding, word2idx, idx2word


#########################
# simple tokenization & vocab
#########################

def build_vocab_from_word2idx(word2idx):
    vocab = {w: idx for w, idx in word2idx.items()}
    return vocab

def split_sentences(text):
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]


def cross_merge(rationale_text, suprious_text, noise_text):
    rationale_sents = split_sentences(rationale_text)
    suprious_sents = split_sentences(suprious_text)
    noise_sents = split_sentences(noise_text)

    merged_sents = []
    rationale_positions = []
    i = j = k = 0
    current_idx = 0
    
    while i < len(rationale_sents) or j < len(suprious_sents) or k < len(noise_sents):
        if i < len(rationale_sents):
            tokens = rationale_sents[i].split()  
            merged_sents.extend(tokens)
            rationale_positions.append((current_idx, current_idx + len(tokens) - 1))
            current_idx += len(tokens)
            i += 1
        if j < len(suprious_sents):
            tokens = suprious_sents[j].split()
            merged_sents.extend(tokens)
            current_idx += len(tokens)
            j += 1
        if k < len(noise_sents):
            tokens = noise_sents[k].split()
            merged_sents.extend(tokens)
            current_idx += len(tokens)
            k += 1
    
    return merged_sents, rationale_positions

def texts_to_tensor(texts,rationales, word2idx, max_len,labels): 
    n = len(texts)
    inputs = np.zeros((n, max_len), dtype=np.int64)
    masks = np.zeros((n, max_len), dtype=np.float32)
    rationale_masks = np.zeros((n, max_len), dtype=np.float32)

    print("The Len of {}".format(len(texts)))
    for i, t in enumerate(texts):
        all_text= texts[i]
        rationale_text = rationales[i]
        label = labels[i]
        suprious_text = all_text.replace(rationale_text, "") 
        noise_text_token = []
        for u,token in enumerate(suprious_text.lower().split()):
            if(u%10==0):
                noise_text_token.append("UNK!")
            noise_text_token.append("UNK")

        noise_text = " ".join(noise_text_token)
        merged_tokens, rationale_pos = cross_merge(rationale_text, suprious_text,noise_text)

        toks = merged_tokens
        if(len(toks)>max_len):
            toks = toks[:max_len]
        else:
            toks = toks
        for j, w in enumerate(toks):
            try:
                inputs[i, j] =  word2idx[w]
                masks[i, j] = 1.
            except:

                if("UNK" in w):
                    inputs[i, j] = 0.
                    masks[i, j] = 1.
                else:
                    inputs[i, j] = 0.
                    masks[i, j] = 1.


        # construct rationale
        for zs in rationale_pos: 
            start = zs[0]
            end = zs[1]
            if start >= max_len:
                continue
            if end >= max_len:
                end = max_len
            for idx in range(start, end):
                rationale_masks[i][idx] = 1 

    return torch.from_numpy(inputs), torch.from_numpy(masks),torch.from_numpy(rationale_masks)

def texts_to_tensor_teacher(texts,rationales, word2idx, max_len,labels): 
    n = len(texts)
    inputs = np.zeros((n, max_len), dtype=np.int64)
    masks = np.zeros((n, max_len), dtype=np.float32)

    print("The Len of {}".format(len(texts)))
    for i, t in enumerate(texts):
        rationale_text = rationales[i]
        toks = rationale_text.lower().split()

        if(len(toks)>max_len):
            toks = toks[:max_len]
        else:
            toks = toks
        for j, w in enumerate(toks):
            try:
                inputs[i, j] =  word2idx[w]
                masks[i, j] = 1.
            except:
                if("UNK" in w):
                    inputs[i, j] = 0.
                    masks[i, j] = 1.
                else:
                    inputs[i, j] = 0.
                    masks[i, j] = 1.

    return torch.from_numpy(inputs), torch.from_numpy(masks)

#########################
# Regularizers
#########################

def get_sparsity_loss(z_mask, masks, target_sparsity):
    selected = (z_mask * masks).sum(dim=1) / (masks.sum(dim=1) + 1e-12)
    loss = F.mse_loss(selected, torch.full_like(selected, target_sparsity))
    return loss

def get_continuity_loss(z_mask):
    diff = torch.abs(z_mask[:, 1:] - z_mask[:, :-1])
    return diff.mean()


#########################
# Training script
#########################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lm_feature', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--sparsity_percentage', type=float, default=0.6)
    parser.add_argument('--temperature', type=float, default=2.0)        
    parser.add_argument('--cls_lambda', type=float, default=1)
    parser.add_argument('--sparsity_lambda', type=float, default=1000.0)
    parser.add_argument('--distribution_lambda', type=float, default=0.1)
    parser.add_argument('--continuity_lambda', type=float, default=10.0)
    parser.add_argument('--gpu', type=str, default=3)
    parser.add_argument('--save_path', type=str, default='./tag_rnp_ckpt.pt')
    parser.add_argument('--seed',type=int,default=12252018, help='The aspect number of beer review [20226666,12252018]')

    parser.add_argument('--embedding_dir', type=str, default='./data/glove/embeddings')
    parser.add_argument('--embedding_name', type=str, default='glove.6B.100d.txt')

    parser.add_argument('--graph_backbone', type=str, default='MLP') # GCN / SAGE / GIN / RevGAT /MLP
    parser.add_argument('--gnn_layers', type=int, default=2) 

    # Dataset
    parser.add_argument('--data_type', type=str, default='cora') #cora pubmed products arxiv_2023
    parser.add_argument('--visual_interval', type=int, default=100) 

    return parser.parse_args()


def evaluate(model, data, inputs, masks,rationale_masks, device):
    model.eval()

    train_num_true_pos = 0.
    train_num_predicted_pos = 0.
    train_num_real_pos = 0.
    train_num_words = 0

    val_num_true_pos = 0.
    val_num_predicted_pos = 0.
    val_num_real_pos = 0.
    val_num_words = 0

    test_num_true_pos = 0.
    test_num_predicted_pos = 0.
    test_num_real_pos = 0.
    test_num_words = 0


    # move data and tensors
    inputs = inputs.to(device)
    masks = masks.to(device)
    rationale_masks = rationale_masks.to(device)
    data = data.to(device)

    with torch.no_grad():
        z, logits = model(inputs=inputs, masks=masks, edge_index=data.edge_index)
        pred = logits.argmax(dim=1).cpu()
        y = data.y.cpu()

        acc_train = ((pred[data.train_mask] == y[data.train_mask]).float().mean().item())
        acc_val = ((pred[data.val_mask] == y[data.val_mask]).float().mean().item())
        acc_test = ((pred[data.test_mask] == y[data.test_mask]).float().mean().item())


        pred_rationale = z[:, :, 1]*masks
        train_num_true_pos_, train_num_predicted_pos_, train_num_real_pos_ = compute_micro_stats(rationale_masks[data.train_mask], pred_rationale[data.train_mask])
        train_num_true_pos += train_num_true_pos_
        train_num_predicted_pos += train_num_predicted_pos_
        train_num_real_pos += train_num_real_pos_
        train_num_words += torch.sum(masks[data.train_mask])


        val_num_true_pos_, val_num_predicted_pos_, val_num_real_pos_ = compute_micro_stats(rationale_masks[data.val_mask], pred_rationale[data.val_mask])
        val_num_true_pos += val_num_true_pos_
        val_num_predicted_pos += val_num_predicted_pos_
        val_num_real_pos += val_num_real_pos_
        val_num_words += torch.sum(masks[data.val_mask])

        test_num_true_pos_, test_num_predicted_pos_, test_num_real_pos_ = compute_micro_stats(rationale_masks[data.test_mask], pred_rationale[data.test_mask])
        test_num_true_pos += test_num_true_pos_
        test_num_predicted_pos += test_num_predicted_pos_
        test_num_real_pos += test_num_real_pos_
        test_num_words += torch.sum(masks[data.test_mask])

    train_micro_precision = train_num_true_pos / train_num_predicted_pos
    train_micro_recall = train_num_true_pos / train_num_real_pos
    train_micro_f1 = 2 * (train_micro_precision * train_micro_recall) / (train_micro_precision + train_micro_recall)
    train_sparsity = train_num_predicted_pos / train_num_words

    val_micro_precision = val_num_true_pos / val_num_predicted_pos
    val_micro_recall = val_num_true_pos / val_num_real_pos
    val_micro_f1 = 2 * (val_micro_precision * val_micro_recall) / (val_micro_precision + val_micro_recall)
    val_sparsity = val_num_predicted_pos / val_num_words


    test_micro_precision = test_num_true_pos / test_num_predicted_pos
    test_micro_recall = test_num_true_pos / test_num_real_pos
    test_micro_f1 = 2 * (test_micro_precision * test_micro_recall) / (test_micro_precision + test_micro_recall)
    test_sparsity = test_num_predicted_pos / test_num_words


    return acc_train, acc_val, acc_test, (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
    (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1)



def evaluate_for(model, data, inputs, masks,rationale_masks, device,lm_feature=None):
    model.eval()

    train_num_true_pos = 0.
    train_num_predicted_pos = 0.
    train_num_real_pos = 0.
    train_num_words = 0

    val_num_true_pos = 0.
    val_num_predicted_pos = 0.
    val_num_real_pos = 0.
    val_num_words = 0

    test_num_true_pos = 0.
    test_num_predicted_pos = 0.
    test_num_real_pos = 0.
    test_num_words = 0


    # move data and tensors
    inputs = inputs.to(device)
    masks = masks.to(device)
    rationale_masks = rationale_masks.to(device)
    data = data.to(device)



    results_list = []
    with torch.no_grad():
        z, logits,z_list = model(inputs=inputs, masks=masks,lm_feature=lm_feature, edge_index=data.edge_index,is_evaluating=True)
        pred = logits.argmax(dim=1).cpu()
        y = data.y.cpu()

        acc_train = ((pred[data.train_mask] == y[data.train_mask]).float().mean().item())
        acc_val = ((pred[data.val_mask] == y[data.val_mask]).float().mean().item())
        acc_test = ((pred[data.test_mask] == y[data.test_mask]).float().mean().item())


        new_z_list = [z] + z_list

        for z in new_z_list:
            pred_rationale = z[:, :, 1]*masks
            train_num_true_pos_, train_num_predicted_pos_, train_num_real_pos_ = compute_micro_stats(rationale_masks[data.train_mask], pred_rationale[data.train_mask])
            train_num_true_pos += train_num_true_pos_
            train_num_predicted_pos += train_num_predicted_pos_
            train_num_real_pos += train_num_real_pos_
            train_num_words += torch.sum(masks[data.train_mask])


            val_num_true_pos_, val_num_predicted_pos_, val_num_real_pos_ = compute_micro_stats(rationale_masks[data.val_mask], pred_rationale[data.val_mask])
            val_num_true_pos += val_num_true_pos_
            val_num_predicted_pos += val_num_predicted_pos_
            val_num_real_pos += val_num_real_pos_
            val_num_words += torch.sum(masks[data.val_mask])

            test_num_true_pos_, test_num_predicted_pos_, test_num_real_pos_ = compute_micro_stats(rationale_masks[data.test_mask], pred_rationale[data.test_mask])
            test_num_true_pos += test_num_true_pos_
            test_num_predicted_pos += test_num_predicted_pos_
            test_num_real_pos += test_num_real_pos_
            test_num_words += torch.sum(masks[data.test_mask])

            train_micro_precision = train_num_true_pos / train_num_predicted_pos
            train_micro_recall = train_num_true_pos / train_num_real_pos
            train_micro_f1 = 2 * (train_micro_precision * train_micro_recall) / (train_micro_precision + train_micro_recall)
            train_sparsity = train_num_predicted_pos / train_num_words

            val_micro_precision = val_num_true_pos / val_num_predicted_pos
            val_micro_recall = val_num_true_pos / val_num_real_pos
            val_micro_f1 = 2 * (val_micro_precision * val_micro_recall) / (val_micro_precision + val_micro_recall)
            val_sparsity = val_num_predicted_pos / val_num_words


            test_micro_precision = test_num_true_pos / test_num_predicted_pos
            test_micro_recall = test_num_true_pos / test_num_real_pos
            test_micro_f1 = 2 * (test_micro_precision * test_micro_recall) / (test_micro_precision + test_micro_recall)
            test_sparsity = test_num_predicted_pos / test_num_words


            cur_result = (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
        (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1)
            
            results_list.append(cur_result)
    
    return acc_train, acc_val, acc_test, results_list


def evaluate_teacher(model, data, inputs, masks,rationale_masks, device,idx2word):
    model.eval()

    # move data and tensors
    inputs = inputs.to(device)
    masks = masks.to(device)
    rationale_masks = rationale_masks.to(device)
    data = data.to(device)

    with torch.no_grad():
        logits = model(inputs, masks, data.edge_index)
        pred = logits.argmax(dim=1).cpu()
        y = data.y.cpu()

        acc_train = ((pred[data.train_mask] == y[data.train_mask]).float().mean().item())
        acc_val = ((pred[data.val_mask] == y[data.val_mask]).float().mean().item())
        acc_test = ((pred[data.test_mask] == y[data.test_mask]).float().mean().item())


    return acc_train, acc_val, acc_test

def main(args):
    print("SEED:{}".format(seed))
    best_result = {}

    torch.manual_seed(args.seed)
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    ######################
    # device
    ######################
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    # use TAPE explanation
    use_gpt_explanation = True 
    #####################
    # Dataset arguments
    #####################
    print("Reading Dataset...")
    if(args.data_type=="cora"):
        data, texts,rationales,gpt_explanation = get_raw_text_cora(use_text=True, seed=args.seed,use_gpt_explanation=use_gpt_explanation)
    elif(args.data_type=="pubmed"):
        data, texts,rationales,gpt_explanation = get_raw_text_pubmed_subgraph_aligned(use_text=True, seed=args.seed,use_gpt_explanation=use_gpt_explanation)
    elif(args.data_type=="products"):
        data, texts,rationales,gpt_explanation = get_raw_text_products_subgraph_aligned(use_text=True, seed=args.seed,use_gpt_explanation=use_gpt_explanation)
    elif(args.data_type=="arxiv_2023"):
        data, texts,rationales,gpt_explanation = get_raw_text_arxiv_2023_subgraph_aligned(use_text=True, seed=args.seed,use_gpt_explanation=use_gpt_explanation)
    else:
        pass
        sys.exit(1)
    print('num nodes', data.num_nodes)

    # load glove embedding
    glove_path = os.path.join(args.embedding_dir, args.embedding_name)
    pretrained_embedding, word2idx,idx2word = get_glove_embedding(glove_path)
    vocab_size = len(word2idx)
    print('vocab size', vocab_size)

    inputs, masks,rationale_masks = texts_to_tensor(texts,rationales, word2idx, args.max_len,data.y)
    if use_gpt_explanation:
        teacher_inputs, teacher_masks = texts_to_tensor_teacher(texts,gpt_explanation, word2idx, args.max_len,data.y) #gpt解释
    else:
        teacher_inputs, teacher_masks = texts_to_tensor_teacher(texts,rationales, word2idx, args.max_len,data.y)  #原文

    dataset = "ogbn-products" if args.data_type=="products" else args.data_type
    tape_root = "TAPE/Embdeeing/" # TAPE_Embedding_Path
    plm_embedding = "TAPE/Embedding_FT/prt_lm/{}2/microsoft/deberta-base-seed{}.emb".format(dataset,dataset,args.seed)
    LM_emb_path = os.path.join(tape_root,plm_embedding)

    num_nodes = data.y.shape[0]
    lm_feature = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
    lm_feature = lm_feature.to(device)

    if(args.lm_feature==0):
        lm_feature = None

    # prepare model args
    class A: pass
    A.vocab_size = vocab_size
    A.embedding_dim = args.embedding_dim
    A.pretrained_embedding = pretrained_embedding
    A.hidden_dim = args.hidden_dim
    A.num_layers = args.num_layers
    A.dropout = args.dropout
    A.num_class = int(data.y.max().item() + 1)
    A.graph_backbone = args.graph_backbone
    A.gnn_layers = args.gnn_layers
    A.max_len = args.max_len
    A.lm_feature_dim = lm_feature.shape[1] if lm_feature!=None else args.hidden_dim
    args_model = A

    model = StudentTagRnpModel2(args_model)
    model = model.to(device)

    g_para=[]
    for p in model.generator.parameters():
        if p.requires_grad==True:
            g_para.append(p)
    p_para=[]
    for p in model.gnn_layers.parameters():
        if p.requires_grad==True:
            p_para.append(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    teacher_model = TeacherEncoder(args_model)
    teacher_model = teacher_model.to(device)

    teacher_para=[
        {'params': teacher_model.embedding_layer.parameters(), 'lr':args.lr},
        {'params': teacher_model.cls_teacher.parameters(), 'lr':args.lr},
        {'params': teacher_model.gnn_layers.parameters(), 'lr':args.lr},
    ]
    teacher_optimizer = torch.optim.Adam(teacher_para, lr=args.lr)

    for p in model.rationale_decoder.parameters():
        p.requires_grad = False

    for p in model.prob_classifier.parameters():
        p.requires_grad = False
        
    # move data and tensors
    inputs = inputs.to(device)
    masks = masks.to(device)
    rationale_masks = rationale_masks.to(device)
    data = data.to(device)

    teacher_inputs, teacher_masks = teacher_inputs.to(device), teacher_masks.to(device)


    ####Train TeacherEncoder #####
    for epoch in tqdm(range(1, 1*(args.epochs + 1))):
        start = time.time()
        teacher_model.train()
        teacher_optimizer.zero_grad()

        teacher_logits = teacher_model(teacher_inputs, teacher_masks, data.edge_index) 
        cls_loss = args.cls_lambda * F.nll_loss(teacher_logits[data.train_mask], data.y[data.train_mask])

        loss = cls_loss
        loss.backward()
        teacher_optimizer.step()

        end = time.time()


        if epoch % 1 == 0 or epoch == 1:
            acc_train, acc_val, acc_test = \
            evaluate_teacher(teacher_model, data, teacher_inputs, teacher_masks,rationale_masks, device,idx2word)

            print(f"Teacher Epoch #{epoch:03d}: Time:{end-start} s")
            print(f"Loss {loss.item():.4f} CLS {cls_loss.item():.4f}")
            print(f"Train {acc_train:.4f} Val {acc_val:.4f} Test {acc_test:.4f}")


    ####Train Students ####
    ##################
    best_val = 0.
    best_test = 0.
    best_epoch = 0

    f1_best_val = [0]
    best_val_epoch = [0]
    acc_best_val = [0]

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        teacher_model.eval()
        
        torch.autograd.set_detect_anomaly(True)

        z, student_logits,output_list = model(inputs=inputs, masks=masks, edge_index=data.edge_index) 
        cls_loss = args.cls_lambda * F.nll_loss(student_logits[data.train_mask], data.y[data.train_mask])
        z_sel = z[:, :, 1]
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(z_sel[data.train_mask], masks[data.train_mask], args.sparsity_percentage) #[data.train_mask]
        continuity_loss = args.continuity_lambda * get_continuity_loss(z_sel[data.train_mask]) # [data.train_mask]


        teacher_logits = teacher_model(inputs, z_sel, data.edge_index).detach()
        T = args.temperature  
            
        teacher_probs = F.softmax(teacher_logits[data.train_mask] / T, dim=1)  
        student_log_probs = F.log_softmax(student_logits[data.train_mask] / T, dim=1) 

        distillation_loss_1 = F.kl_div(
                student_log_probs, 
                teacher_probs,
                reduction="batchmean"
            ) * (T * T)  

        student_probs = F.softmax(student_logits[data.train_mask] / T, dim=1)  
        teacher_log_probs = F.log_softmax(teacher_logits[data.train_mask] / T, dim=1)  

        distillation_loss_2 = F.kl_div(
                teacher_log_probs, 
                student_probs,
                reduction="batchmean"
            ) * (T * T)  

        distillation_loss = 0.5*distillation_loss_1 + 0.5*distillation_loss_2

        layer_distribution_loss = 0.0
        for cur_student_logits in output_list:
            cur_student_log_probs = F.log_softmax(cur_student_logits[data.train_mask] / T, dim=1) 

            cur_layer_distillation_loss_1 = F.kl_div(
                cur_student_log_probs, 
                teacher_probs,
                reduction="batchmean"
            ) * (T * T)  


            cur_student_probs = F.softmax(cur_student_logits[data.train_mask] / T, dim=1)  # teacher 分布（soft）
            cur_layer_distillation_loss_2 = F.kl_div(
                    teacher_log_probs, 
                    cur_student_probs,
                    reduction="batchmean"
                ) * (T * T)  

            layer_distribution_loss = layer_distribution_loss + 1.0*cur_layer_distillation_loss_1+0.0*cur_layer_distillation_loss_2
            
        if(len(output_list)==0):
            pass
        else:
            layer_distribution_loss = layer_distribution_loss/len(output_list)


        distribution_loss =  cls_loss + args.distribution_lambda* distillation_loss + args.distribution_lambda*layer_distribution_loss

        for p in model.parameters():
            p.requires_grad = True
        for p in model.generator.parameters():
            p.requires_grad = False

        optimizer.zero_grad()
        distribution_loss.backward()
        optimizer.step()



        if epoch % 1 == 0 or epoch == 1:
            acc_train, acc_val, acc_test, results_list = \
            evaluate_for(model, data, inputs, masks,rationale_masks, device)

            print(f"Epoch ##{epoch:03d}:")
            print(f"Loss {loss.item():.4f} CLS {cls_loss.item():.4f} Spar {sparsity_loss.item():.4f} Cont {continuity_loss.item():.4f} Dist {str(distribution_loss)} Lay Dis {str(layer_distribution_loss)}") #
            print(f"Train {acc_train:.4f} Val {acc_val:.4f} Test {acc_test:.4f}")
            
            for u,result in enumerate(results_list):
                (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
            (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1) = result
                
                print("第{}次Evaluating...".format(u))
                print("Train dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1))
                print("validation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1))
                print("annotation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1))

            (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
            (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1) = results_list[0]
            print("Train dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1))
            print("validation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1))
            print("annotation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1))


    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()

        torch.autograd.set_detect_anomaly(True)

        z, student_logits,output_list = model(inputs=inputs, masks=masks, edge_index=data.edge_index,lm_feature=lm_feature) 
        cls_loss = args.cls_lambda * F.nll_loss(student_logits[data.train_mask], data.y[data.train_mask])
        z_sel = z[:, :, 1]
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(z_sel[data.train_mask], masks[data.train_mask], args.sparsity_percentage) #[data.train_mask]
        continuity_loss = args.continuity_lambda * get_continuity_loss(z_sel[data.train_mask]) # [data.train_mask

        loss_s = cls_loss + sparsity_loss + continuity_loss

        if(lm_feature!=None):
            for p in model.parameters():
                p.requires_grad = True
            for p in model.generator.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.generator.parameters():
                p.requires_grad = True
        optimizer.zero_grad()
        loss_s.backward()
        optimizer.step()


        if (args.visual_interval and ((epoch + 1) % args.visual_interval in [0, 1]) and epoch != 0):
            import random
            sample_id = 328 
            labels = data.y.cpu()

            print('epoch # %d: sample id: %d, true label: %d' % (epoch + 1, sample_id, labels[sample_id]))
        
            from visualize import show_binary_rationale
            show_binary_rationale(inputs[sample_id, :].detach().cpu().numpy(),masks[sample_id].detach().cpu().numpy(),idx2word)
            show_binary_rationale(inputs[sample_id, :].detach().cpu().numpy(),rationale_masks[sample_id].detach().cpu().numpy(),idx2word)
            show_binary_rationale(inputs[sample_id, :].detach().cpu().numpy(),z[sample_id, :, 1].detach().cpu().numpy(),idx2word)



        if epoch % 1 == 0 or epoch == 1:
            acc_train, acc_val, acc_test, results_list = \
            evaluate_for(model, data, inputs, masks,rationale_masks, device,lm_feature=lm_feature)


            print(f"Epoch #{epoch:03d}:")
            print(f"Loss {loss.item():.4f} CLS {cls_loss.item():.4f} Spar {sparsity_loss.item():.4f} Cont {continuity_loss.item():.4f} ") #Dist {distribution_loss.item():.4f}  Lay Dis {layer_distribution_loss.item():.4f}
            print(f"Train {acc_train:.4f} Val {acc_val:.4f} Test {acc_test:.4f}")
            

            for u,result in enumerate(results_list):
                (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
            (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1) = result
                
                print("第{}次Evaluating...".format(u))

                print("Train dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1))
                print("validation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1))
                print("annotation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1))


            (train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1),\
            (val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1),(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1) = results_list[0]
            print("Train dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(train_sparsity,train_micro_precision, train_micro_recall,train_micro_f1))
            print("validation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(val_sparsity,val_micro_precision, val_micro_recall,val_micro_f1))
            print("annotation dataset : sparsity:{} precision:{:.4f} recall:{:.4f} f1-score:{:.4f}"
                    .format(test_sparsity,test_micro_precision, test_micro_recall,test_micro_f1))

            current_epoch_result = {
                "sparsity_percentage":args.sparsity_percentage,
                "vali_s":val_sparsity,
                "vali_acc":acc_val,
                "vali_rat_p":val_micro_precision,
                "vali_rat_r":val_micro_recall,
                "vali_rat_f1":val_micro_f1,
                
                
                "test_s":test_sparsity,
                "test_acc":acc_test,
                "test_rat_p":test_micro_precision,
                "test_rat_r":test_micro_recall,
                "test_rat_f1":test_micro_f1,
                
            }

            if acc_val > best_val:
                best_val = acc_val
                torch.save(model.state_dict(), args.save_path)
                best_epoch = epoch

                acc_best_val.append(best_val)
                best_val_epoch.append(epoch)
                f1_best_val.append(test_micro_f1)

                best_result = current_epoch_result
                best_result["best_epoch"] = best_epoch

    print(acc_best_val)
    print(best_val_epoch)
    print(f1_best_val)

    best_result["help"] = "SEED {} Experiment".format(args.seed)

    return best_result


if __name__ == '__main__':
    #####################
    # parse arguments
    #####################
    args = parse_args()

    seeds = [0,1,2]

    all_result = []
    start = time.time()
    for seed in seeds:
        args.seed = seed
        result = main(args)
        all_result.append(result)
    end = time.time()




    if len(all_result) > 1 or len(all_result) == 1:
        df = pd.DataFrame(all_result)
        exclude = {"best_epoch", "help"}
        cols = [c for c in df.columns if c not in exclude]
        means = df[cols].mean()
        stds = df[cols].std()

        for i,results in enumerate(all_result):
            print(i)
            print(result)
            print("\n")

        for key in cols:
            print(f"{key}: mean={means[key]:.4f}, std={stds[key]:.4f}")
    
    print(f"\n Running time: {round((end-start)/len(seeds), 2)}s")

    
