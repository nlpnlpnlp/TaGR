import numpy as np
import torch
import random
import json
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from tqdm import tqdm


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    return data, data_citeid

def get_cora_casestudy1(SEED=0):
    '''inductive setting'''
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # load data
    data_name = 'cora'
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id   = np.sort(node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id  = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask   = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask  = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

    edge_index = data.edge_index.numpy()

    train_nodes = set(data.train_id.tolist())
    val_nodes   = set(data.val_id.tolist())
    test_nodes  = set(data.test_id.tolist())

    mask = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if (u in train_nodes and v in train_nodes) \
           or (u in val_nodes and v in val_nodes) \
           or (u in test_nodes and v in test_nodes):
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    data.edge_index = torch.tensor(edge_index[:, mask]).long()

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun

def parse_cora():
    path = 'dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()

def get_raw_text_cora(use_text=False, seed=0,use_gpt_explanation=False):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('./dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = './responses/spurious/cora.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0
    with open(path, 'r',encoding='utf-8') as file:
        json_data = json.load(file)

        for i,json_example in enumerate(tqdm(json_data)):
            title = json_example['title']
            abstract = json_example['abstract']

            noise = json_example['llm_response']['content_2_round']
            cur_text = str(noise+'\n'+title+'\n'+abstract)
            cur_rationale = str(title+'\n'+abstract)

            cur_text = str(title+'\n'+abstract+'\n'+noise)
            cur_rationale = str(title+'\n'+abstract)
            text.append(cur_text)
            rationale.append(cur_rationale)

            cur_text = cur_text.lower().split()
            cur_rationale = cur_rationale.lower().split()

            if(len(cur_text)>text_max_len):
                text_max_len = len(cur_text)
            if(len(cur_rationale)>rational_max_len):
                rational_max_len = len(cur_rationale)

            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)
    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(text_max_len,rational_max_len,text_tot_len/len(json_data),rational_tot_len/len(json_data),rational_tot_len/text_tot_len))
    
    num_edges = data.edge_index.size(1)

    if use_gpt_explanation:
        import os
        dataset = 'cora'
        folder_path = 'responses/explanation/{}'.format(dataset)
        print(f"using gpt explanation: {folder_path}")
        n = data.y.shape[0]
        gpt_explanation = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                gpt_explanation.append(content)
        
        return data, text, rationale,gpt_explanation
    
    
    return data, text, rationale

def get_raw_text_cora1(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
        print(lines)

    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

        print(pid,fn)
        print(pid_filename)
        print('\n')

    path = 'dataset/cora_orig/mccallum/cora/extractions/'  #52904 files
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)   # extract the title and abstract
    return data, text


def get_raw_text_cora_generate_data(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
        # print(lines)

    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'dataset/cora_orig/mccallum/cora/extractions/'  #52904 files

    citeid_list = []
    text_list = []

    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            # if(pid=="36802"):
            #     print(line)
            if 'Title:' in line:
                title = line
            if 'Abstract:' in line:
                abstract = line
            if 'Author:' in line:
                author = line
            if 'Date:' in line:
                date = line
        text_list.append({
            "citeid":pid,
            "title":title,
            "abstract":abstract,
            "author":author,
            "date":date
        })   # extract the title and abstract

    return data, text_list