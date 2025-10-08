import torch
import pandas as pd
import json

import os
import torch
import json
from torch_geometric.utils import subgraph

def get_raw_text_products(use_text=False, seed=0):
    data = torch.load('./dataset/ogbn_products/ogbn-products_subset.pt')
    print(data.keys()) # ['e_id', 'n_asin', 'n_id', 'x', 'adj_t', 'input_id', 'test_mask', 'batch_size', 'y', 'val_mask', 'train_mask', 'num_nodes']

    data.edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    path = './responses/spurious/ogbn-products.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    from tqdm import tqdm
    with open(path, 'r',encoding='utf-8') as file:
        json_data = json.load(file)

        for json_example in tqdm(json_data):

            title = json_example['product']
            abstract = json_example['description']
            noise = json_example['llm_response']['content_2_round']


            text.append(str(str(noise)+'\n'+str(title)+'\n'+str(abstract)))
            rationale.append(str(title)+'\n'+str(abstract))

            cur_text = str(str(noise)+'\n'+str(title)+'\n'+str(abstract))
            cur_rationale = str(title)+'\n'+str(abstract)
            if(len(cur_text)>text_max_len):
                text_max_len = len(cur_text)
            if(len(cur_rationale)>rational_max_len):
                rational_max_len = len(cur_rationale)

            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)
    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(text_max_len,rational_max_len,text_tot_len/len(json_data),rational_tot_len/len(json_data),rational_tot_len/text_tot_len))
    return data, text, rationale


def get_raw_text_products_subgraph_aligned1(use_text=False, seed=0, num_nodes_sample=5403):
    data = torch.load('./dataset/ogbn_products/ogbn-products_subset.pt')
    data.edge_index = data.adj_t.to_symmetric().coo()
    data.edge_index = torch.stack(data.edge_index[:2], dim=0)

    if num_nodes_sample is not None and num_nodes_sample < data.num_nodes:
        torch.manual_seed(seed)
        train_nodes = torch.nonzero(data.train_mask, as_tuple=True)[0]
        val_nodes = torch.nonzero(data.val_mask, as_tuple=True)[0]
        test_nodes = torch.nonzero(data.test_mask, as_tuple=True)[0]

        n_train = int(num_nodes_sample * len(train_nodes) / data.num_nodes)
        n_val = int(num_nodes_sample * len(val_nodes) / data.num_nodes)
        n_test = num_nodes_sample - n_train - n_val  

        sampled_train = train_nodes[torch.randperm(len(train_nodes))[:n_train]]
        sampled_val = val_nodes[torch.randperm(len(val_nodes))[:n_val]]
        sampled_test = test_nodes[torch.randperm(len(test_nodes))[:n_test]]

        sampled_nodes = torch.cat([sampled_train, sampled_val, sampled_test])
        edge_index, _ = subgraph(sampled_nodes, data.edge_index, relabel_nodes=True)

        data.edge_index = edge_index
        data.x = data.x[sampled_nodes]
        data.y = data.y[sampled_nodes]

        train_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)
        val_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)
        test_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)

        node_id_map = {int(n): i for i, n in enumerate(sampled_nodes)}

        for n in sampled_train:
            train_mask[node_id_map[int(n)]] = True
        for n in sampled_val:
            val_mask[node_id_map[int(n)]] = True
        for n in sampled_test:
            test_mask[node_id_map[int(n)]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data.num_nodes = num_nodes_sample

    if not use_text:
        return data, None, None

    path = './responses/spurious/ogbn-products.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        if num_nodes_sample is not None and num_nodes_sample < data.num_nodes:
            selected_nodes = sampled_nodes.tolist()
        else:
            selected_nodes = list(range(data.num_nodes))

        for n_id in selected_nodes:
            json_example = json_data[n_id]  
            title = json_example['product']
            abstract = json_example['description']
            noise = json_example['llm_response']['content_2_round']

            cur_text = f"{noise}\n{title}\n{abstract}"
            cur_rationale = f"{title}\n{abstract}"

            text.append(cur_text)
            rationale.append(cur_rationale)

            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)
            text_max_len = max(text_max_len, len(cur_text))
            rational_max_len = max(rational_max_len, len(cur_rationale))

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparsity:{}"
          .format(text_max_len, rational_max_len, text_tot_len/len(selected_nodes),
                  rational_tot_len/len(selected_nodes), rational_tot_len/text_tot_len))

    return data, text, rationale


def get_raw_text_products_subgraph_aligned2(use_text=False, seed=0, num_sample_nodes=2701):
    data = torch.load('./dataset/ogbn_products/ogbn-products_subset.pt')
    data.edge_index = data.adj_t.to_symmetric()
    num_nodes = data.num_nodes

    train_nodes = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
    val_nodes = torch.nonzero(data.val_mask, as_tuple=False).view(-1)
    test_nodes = torch.nonzero(data.test_mask, as_tuple=False).view(-1)

    train_sample_num = int(num_sample_nodes * train_nodes.size(0) / num_nodes)
    val_sample_num = int(num_sample_nodes * val_nodes.size(0) / num_nodes)
    test_sample_num = num_sample_nodes - train_sample_num - val_sample_num

    train_sample_nodes = train_nodes[torch.randperm(train_nodes.size(0))[:train_sample_num]]
    val_sample_nodes = val_nodes[torch.randperm(val_nodes.size(0))[:val_sample_num]]
    test_sample_nodes = test_nodes[torch.randperm(test_nodes.size(0))[:test_sample_num]]

    sample_nodes = torch.cat([train_sample_nodes, val_sample_nodes, test_sample_nodes])
    sample_nodes, inverse_idx = torch.sort(sample_nodes)  # ranking to generate masks

    row, col, value = data.edge_index.coo()
    mask = torch.isin(row, sample_nodes) & torch.isin(col, sample_nodes)
    sub_row = row[mask]
    sub_col = col[mask]

    node_id_map = torch.zeros(num_nodes, dtype=torch.long)
    node_id_map[sample_nodes] = torch.arange(sample_nodes.size(0))
    sub_row = node_id_map[sub_row]
    sub_col = node_id_map[sub_col]

    from torch_sparse import SparseTensor
    sub_edge_index = SparseTensor(row=sub_row, col=sub_col, sparse_sizes=(sample_nodes.size(0), sample_nodes.size(0)))

    sub_data = type(data)()  
    sub_data.x = data.x[sample_nodes]
    sub_data.y = data.y[sample_nodes]
    sub_data.edge_index = sub_edge_index
    sub_data.num_nodes = sample_nodes.size(0)

    sub_data.train_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    sub_data.val_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    sub_data.test_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)

    sub_data.train_mask[:train_sample_nodes.size(0)] = True
    sub_data.val_mask[train_sample_nodes.size(0):train_sample_nodes.size(0)+val_sample_nodes.size(0)] = True
    sub_data.test_mask[-test_sample_num:] = True
    data = sub_data
    data.y = data.y.argmax(dim=-1)  

    path = './responses/spurious/ogbn-products.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        selected_nodes = list(range(data.num_nodes))

        for n_id in selected_nodes:
            json_example = json_data[n_id]  
            title = json_example['product']
            abstract = json_example['description']
            noise = json_example['llm_response']['content_2_round']

            cur_text = f"{noise}\n{title}\n{abstract}"
            cur_rationale = f"{title}\n{abstract}"

            text.append(cur_text)
            rationale.append(cur_rationale)

            cur_text = cur_text.lower().split()
            cur_rationale = cur_rationale.lower().split()
            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)
            text_max_len = max(text_max_len, len(cur_text))
            rational_max_len = max(rational_max_len, len(cur_rationale))

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparsity:{}"
          .format(text_max_len, rational_max_len, text_tot_len/len(selected_nodes),
                  rational_tot_len/len(selected_nodes), rational_tot_len/text_tot_len))

    return data, text, rationale


def get_raw_text_products_subgraph_aligned(use_text=False, seed=0, num_sample_nodes=3025,use_gpt_explanation=False):
    import torch
    import random
    import json

    data = torch.load('./dataset/ogbn_products/ogbn-products_subset.pt')

    data.edge_index = data.adj_t.to_symmetric()
    num_nodes = data.num_nodes

    random.seed(seed)
    selected_nodes = random.sample(range(num_nodes), num_sample_nodes)
    selected_nodes.sort()  
    selected_nodes_set = set(selected_nodes)

    sub_x = data.x[selected_nodes] if hasattr(data, 'x') else None
    sub_y = data.y[selected_nodes]

    edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index.to_dense().numpy()
    mask = [(u in selected_nodes_set and v in selected_nodes_set) for u, v in zip(edge_index[0], edge_index[1])]
    mask = torch.tensor(mask, dtype=torch.bool)
    sub_edge_index = data.edge_index[:, mask]

    if isinstance(sub_edge_index, torch.sparse.Tensor):
        sub_edge_index = sub_edge_index.coalesce().indices()  # shape [2, num_edges]
    elif hasattr(sub_edge_index, 'coo'):
        sub_edge_index = sub_edge_index.coo()  # 对于 PyG SparseTensor

    id_map = {nid: i for i, nid in enumerate(selected_nodes)}
    sub_edge_index_remap = torch.tensor([
        [id_map[u.item()] for u in sub_edge_index[0]],
        [id_map[v.item()] for v in sub_edge_index[1]]
    ], dtype=torch.long)

    sub_data = type(data)()  
    sub_data.num_nodes = num_sample_nodes
    sub_data.x = sub_x
    sub_data.y = sub_y.view(-1)  
    sub_data.edge_index = sub_edge_index_remap

    if hasattr(data, 'train_mask'):
        sub_data.train_mask = data.train_mask[selected_nodes]
    if hasattr(data, 'val_mask'):
        sub_data.val_mask = data.val_mask[selected_nodes]
    if hasattr(data, 'test_mask'):
        sub_data.test_mask = data.test_mask[selected_nodes]

    path = './responses/spurious/ogbn-products.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        for n_id in selected_nodes:
            json_example = json_data[n_id]
            title = json_example['product']
            abstract = json_example['description']
            noise = json_example['llm_response']['content_2_round']

            cur_text = f"{noise}\n{title}\n{abstract}"
            cur_rationale = f"{title}\n{abstract}"

            text.append(cur_text)
            rationale.append(cur_rationale)

            cur_text_tokens = cur_text.lower().split()
            cur_rationale_tokens = cur_rationale.lower().split()
            text_tot_len += len(cur_text_tokens)
            rational_tot_len += len(cur_rationale_tokens)
            text_max_len = max(text_max_len, len(cur_text_tokens))
            rational_max_len = max(rational_max_len, len(cur_rationale_tokens))

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparsity:{}"
          .format(text_max_len, rational_max_len, text_tot_len/len(selected_nodes),
                  rational_tot_len/len(selected_nodes), rational_tot_len/text_tot_len))
    num_edges = sub_data.edge_index.size(1)

    if use_gpt_explanation:        
        dataset = 'ogbn-products'
        gpt_explanation = []
        folder_path = 'responses/explanation/{}'.format(dataset)
        print(f"using gpt: {folder_path}")

        all_text = []
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        fild_ids = []
        for i in range(file_count):
            fild_ids.append(i)
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                all_text.append(content)

        index = 0
        for n_id in selected_nodes:
            cur_text = all_text[n_id]
            # print(sub_data.y[index],cur_text)
            # print(n_id,sub_data.y[index],ta_text[index])
            # print("*****",cur_text,"*****")
            gpt_explanation.append(cur_text)
            index = index + 1
        return sub_data, text, rationale,gpt_explanation

    return sub_data, text, rationale


def get_raw_text_products_generate_data(use_text=False, seed=0):
    data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    df = pd.read_csv('dataset/ogbn_products_orig/ogbn-products_subset.csv')

    label_mapping = {}
    label_map_data = pd.read_csv('dataset/ogbn_products/labelidx2productcategory.csv')
    for idx,cate in zip(label_map_data['label idx'], label_map_data['product category']):
        label_mapping[idx] = cate

    data.edge_index = data.adj_t.to_symmetric()
    labels = [item[0] for item in data.y.tolist()]

    if not use_text:
        return data, None
    
    text_list = []

    all_labels = []
    for i,(uid, nid, product, description) in enumerate(zip(df['uid'],df['nid'], df['title'], df['content'])):
        text_list.append({
            "uid":uid,
            "nid":nid,
            "product":product,
            "category":label_mapping[labels[i]],
            "label":labels[i],
            "description":description,
        })   
        all_labels.append(label_mapping[labels[i]])
    
    return data, text_list
