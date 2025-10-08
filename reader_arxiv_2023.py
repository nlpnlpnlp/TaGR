import torch
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
import json

from torch_geometric.data import Data


def get_raw_text_arxiv_2023(use_text=False, seed=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load('dataset/arxiv_2023/graph.pt')
    print(data.keys())

    
    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    print(num_nodes)
    node_id = np.arange(num_nodes)
    print(node_id)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')


    path = './responses/spurious/arxiv_2023.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    with open(path, 'r',encoding='utf-8') as file:
        json_data = json.load(file)

        for json_example in tqdm(json_data):
            title = json_example['title']
            abstract = json_example['abstract']
            noise = json_example['llm_response']['content_2_round']

            cur_text = str(noise)+'\n'+str(title)+'\n'+str(abstract)
            cur_rationale = str(title)+'\n'+str(abstract)
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
    
    return data, text, rationale


def get_raw_text_arxiv_2023_subgraph_aligned1(
    subgraph_nodes=None, use_text=False, seed=0, subgraph_size=1000
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = torch.load("dataset/arxiv_2023/graph.pt")
    num_nodes = len(data.y)
    if subgraph_nodes is None:
        subgraph_nodes = np.random.choice(num_nodes, subgraph_size, replace=False)
    sub_nodes = torch.tensor(sorted(subgraph_nodes), dtype=torch.long)
    sub_nodes_set = set(sub_nodes.tolist())

    mask = [
        (src.item() in sub_nodes_set) and (dst.item() in sub_nodes_set)
        for src, dst in data.edge_index.t()
    ]
    edge_index_filtered = data.edge_index[:, mask]

    data_sub = Data()
    data_sub.x = data.x[sub_nodes] if data.x is not None else None
    data_sub.y = data.y[sub_nodes]
    data_sub.edge_index = edge_index_filtered  
    data_sub.num_nodes = len(sub_nodes)

    data_sub.train_mask = torch.tensor([i in data.train_id for i in sub_nodes])
    data_sub.val_mask = torch.tensor([i in data.val_id for i in sub_nodes])
    data_sub.test_mask = torch.tensor([i in data.test_id for i in sub_nodes])

    if not use_text:
        return data_sub, None

    path = './responses/spurious/arxiv_2023.json'
    text_dict = {}
    rationale_dict = {}

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    from tqdm import tqdm
    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

        for json_example in tqdm(json_data, desc="Loading node texts from JSON"):
            node_id = json_example["node_id"]
            if node_id not in sub_nodes_set:  
                continue

            title = json_example["title"]
            abstract = json_example["abstract"]
            noise = json_example["llm_response"]["content_2_round"]


            cur_text = str(noise) + "\n" + str(title) + "\n" + str(abstract)
            cur_rationale = str(title) + "\n" + str(abstract)

            text_dict[node_id] = cur_text
            rationale_dict[node_id] = cur_rationale

            cur_text = cur_text.lower().split()
            cur_rationale = cur_rationale.lower().split()
            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)

            text_max_len = max(text_max_len, len(cur_text))
            rational_max_len = max(rational_max_len, len(cur_rationale))

    sorted_node_ids = sorted(text_dict.keys())
    text = [text_dict[nid] for nid in sorted_node_ids]
    rationale = [rationale_dict[nid] for nid in sorted_node_ids]


    sub_nodes_set = set(sub_nodes.tolist())

    mask = [(src.item() in sub_nodes_set) and (dst.item() in sub_nodes_set)
            for src, dst in data.edge_index.t()]
    edge_index_filtered = data.edge_index[:, mask]
    old2new = {old.item(): new for new, old in enumerate(sub_nodes)}

    src_list = [old2new[int(src)] for src in edge_index_filtered[0]]
    dst_list = [old2new[int(dst)] for dst in edge_index_filtered[1]]
    edge_index_relabel = torch.tensor([src_list, dst_list], dtype=torch.long)

    data_sub.edge_index = edge_index_relabel
    data_sub.num_nodes = len(sub_nodes)

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(text_max_len,rational_max_len,text_tot_len/len(sorted_node_ids),rational_tot_len/len(sorted_node_ids),rational_tot_len/text_tot_len))
    return data_sub, text, rationale

def get_raw_text_arxiv_2023_subgraph_aligned2(
    subgraph_nodes=None, use_text=False, seed=0, subgraph_size=1000
):
    from torch_geometric.data import Data
    import json
    from tqdm import tqdm
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = torch.load("dataset/arxiv_2023/graph.pt")
    num_nodes = len(data.y)

    if subgraph_nodes is None:
        n_train = int(subgraph_size * 0.6)
        n_val   = int(subgraph_size * 0.2)
        n_test  = subgraph_size - n_train - n_val

        train_nodes = np.random.choice(data.train_id, size=n_train, replace=False)
        val_nodes = np.random.choice(data.train_id, size=n_val, replace=False)
        test_nodes = np.random.choice(data.train_id, size=n_test, replace=False)

        subgraph_nodes = np.concatenate([train_nodes, val_nodes, test_nodes])

    sub_nodes = torch.tensor(sorted(subgraph_nodes), dtype=torch.long)
    sub_nodes_set = set(sub_nodes.tolist())

    mask = [
        (src.item() in sub_nodes_set) and (dst.item() in sub_nodes_set)
        for src, dst in data.edge_index.t()
    ]
    edge_index_filtered = data.edge_index[:, mask]

    data_sub = Data()
    data_sub.x = data.x[sub_nodes] if data.x is not None else None
    data_sub.y = data.y[sub_nodes]
    old2new = {old.item(): new for new, old in enumerate(sub_nodes)}
    src_list = [old2new[int(src)] for src in edge_index_filtered[0]]
    dst_list = [old2new[int(dst)] for dst in edge_index_filtered[1]]
    edge_index_relabel = torch.tensor([src_list, dst_list], dtype=torch.long)

    data_sub.edge_index = edge_index_relabel
    data_sub.num_nodes = len(sub_nodes)

    data_sub.train_mask = torch.tensor([i in data.train_id for i in sub_nodes])
    data_sub.val_mask   = torch.tensor([i in data.val_id for i in sub_nodes])
    data_sub.test_mask  = torch.tensor([i in data.test_id for i in sub_nodes])

    if not use_text:
        return data_sub, None, None

    path = './responses/spurious/arxiv_2023.json'
    text_dict = {}
    rationale_dict = {}

    text_tot_len, rational_tot_len = 0.0, 0.0
    text_max_len, rational_max_len = 0.0, 0.0

    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
        for json_example in tqdm(json_data, desc="Loading node texts from JSON"):
            node_id = json_example["node_id"]
            if node_id not in sub_nodes_set:
                continue

            title = json_example["title"]
            abstract = json_example["abstract"]
            noise = json_example["llm_response"]["content_2_round"]

            cur_text = str(noise) + "\n" + str(title) + "\n" + str(abstract)
            cur_rationale = str(title) + "\n" + str(abstract)

            text_dict[node_id] = cur_text
            rationale_dict[node_id] = cur_rationale

            cur_text_len = len(cur_text.lower().split())
            cur_rationale_len = len(cur_rationale.lower().split())
            text_tot_len += cur_text_len
            rational_tot_len += cur_rationale_len
            text_max_len = max(text_max_len, cur_text_len)
            rational_max_len = max(rational_max_len, cur_rationale_len)

    sorted_node_ids = sorted(text_dict.keys())
    text = [text_dict[nid] for nid in sorted_node_ids]
    rationale = [rationale_dict[nid] for nid in sorted_node_ids]

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparsity:{}".format(
        text_max_len, rational_max_len,
        text_tot_len/len(sorted_node_ids),
        rational_tot_len/len(sorted_node_ids),
        rational_tot_len/text_tot_len
    ))

    return data_sub, text, rationale


def get_raw_text_arxiv_2023_subgraph_aligned(
    subgraph_nodes=None, use_text=False, seed=0, subgraph_size=2619,
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,use_gpt_explanation=False
):

    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = torch.load("dataset/arxiv_2023/graph.pt")
    num_nodes = len(data.y)

    if subgraph_nodes is None:
        n_train = min(int(subgraph_size * train_ratio), len(data.train_id))
        n_val   = min(int(subgraph_size * val_ratio), len(data.val_id))
        n_test  = min(subgraph_size - n_train - n_val, len(data.test_id))

        train_nodes = np.random.choice(data.train_id, n_train, replace=False)
        val_nodes   = np.random.choice(data.val_id, n_val, replace=False)
        test_nodes  = np.random.choice(data.test_id, n_test, replace=False)

        subgraph_nodes = np.concatenate([train_nodes, val_nodes, test_nodes])

    sub_nodes = torch.tensor(sorted(subgraph_nodes), dtype=torch.long)
    sub_nodes_set = set(sub_nodes.tolist())

    mask = [(src.item() in sub_nodes_set and dst.item() in sub_nodes_set)
            for src, dst in data.edge_index.t()]
    edge_index_filtered = data.edge_index[:, mask]

    old2new = {old.item(): new for new, old in enumerate(sub_nodes)}
    src_list = [old2new[int(src)] for src in edge_index_filtered[0]]
    dst_list = [old2new[int(dst)] for dst in edge_index_filtered[1]]
    edge_index_relabel = torch.tensor([src_list, dst_list], dtype=torch.long)

    data_sub = Data()
    data_sub.x = data.x[sub_nodes] if data.x is not None else None
    data_sub.y = data.y[sub_nodes]
    data_sub.edge_index = edge_index_relabel
    data_sub.num_nodes = len(sub_nodes)

    train_id_set = set(data.train_id.tolist())
    val_id_set   = set(data.val_id.tolist())
    test_id_set  = set(data.test_id.tolist())

    data_sub.train_mask = torch.tensor([nid.item() in train_id_set for nid in sub_nodes], dtype=torch.bool)
    data_sub.val_mask   = torch.tensor([nid.item() in val_id_set for nid in sub_nodes], dtype=torch.bool)
    data_sub.test_mask  = torch.tensor([nid.item() in test_id_set for nid in sub_nodes], dtype=torch.bool)

    text, rationale = None, None
    if use_text:
        path = './responses/spurious/arxiv_2023.json'
        text_dict = {}
        rationale_dict = {}

        text_tot_len = 0.0
        rational_tot_len = 0.0
        text_max_len = 0.0
        rational_max_len = 0.0
        with open(path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            for json_example in tqdm(json_data, desc="Loading node texts from JSON"):
                node_id = json_example["node_id"]
                if node_id not in sub_nodes_set:
                    continue
                title = json_example["title"]
                abstract = json_example["abstract"]
                noise = json_example["llm_response"]["content_2_round"]

                text_dict[node_id] = f"{title}\n{abstract}\n{noise}"
                rationale_dict[node_id] = f"{title}\n{abstract}"

                cur_text = f"{title}\n{abstract}\n{noise}"
                cur_rationale = f"{title}\n{abstract}"
                if(len(cur_text)>text_max_len):
                    text_max_len = len(cur_text)
                if(len(cur_rationale)>rational_max_len):
                    rational_max_len = len(cur_rationale)

                text_tot_len += len(cur_text)
                rational_tot_len += len(cur_rationale)

        print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(text_max_len,rational_max_len,text_tot_len/len(json_data),rational_tot_len/len(json_data),rational_tot_len/text_tot_len))

        sorted_node_ids = sorted(text_dict.keys())
        text = [text_dict[nid] for nid in sorted_node_ids]
        rationale = [rationale_dict[nid] for nid in sorted_node_ids]

    data_sub.node_id = sub_nodes  # 保存原始节点 id 对齐文本
    num_edges = data_sub.edge_index.size(1)

    import os
    if use_gpt_explanation:
        text_dict = {}
        dataset = 'arxiv_2023'
        folder_path = 'responses/explanation/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]

        gpt_explanation = []
        for i in range(n):
            node_id = i
            if node_id not in sub_nodes_set:
                continue

            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text_dict[node_id] = content
        
        sorted_node_ids_tmp = sorted(text_dict.keys())
        gpt_explanation = [text_dict[nid] for nid in sorted_node_ids_tmp]

        return data_sub, text, rationale, gpt_explanation

    return data_sub, text, rationale


def get_raw_text_arxiv_2023_generate_data(use_text=False, seed=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load('dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')
    text_list = []
    for arxiv_id,ti, ab,category,label,node_id in zip(df['arxiv_id'], df['title'], df['abstract'],df['category'],df['label'],df['node_id']):
        text_list.append({
            "arxiv_id":arxiv_id,
            "title":ti,
            "abstract":ab,
            "category":category,
            "label":label,
            "node_id":node_id,
        })   # extract the title and abstract
    return data, text_list