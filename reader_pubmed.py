import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
import os

import torch
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.preprocessing import normalize

import json
from tqdm import tqdm

'''
return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs
'''

def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt('pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_pubid


def get_pubmed_casestudy_subgraph(corrected=False, SEED=0, num_nodes_sample=1972):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()

    data_X = np.array(data_X, dtype=np.float32)
    data_Y = np.array(data_Y, dtype=np.int64)
    data_pubid = np.array(data_pubid)
    data_edges = np.array(data_edges)

    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    all_node_ids = np.arange(len(data_X))

    sampled_nodes = np.random.choice(all_node_ids, size=min(num_nodes_sample, len(all_node_ids)), replace=False)
    sampled_nodes = np.array(sampled_nodes, dtype=int)

    mask = np.isin(data_edges[0], sampled_nodes) & np.isin(data_edges[1], sampled_nodes)
    sampled_edges = data_edges[:, mask]

    old_to_new = {old: new for new, old in enumerate(sampled_nodes)}
    sampled_edges_mapped = np.array([[old_to_new[i] for i in sampled_edges[0]],
                                     [old_to_new[i] for i in sampled_edges[1]]])

    # PyG Data
    data = Data(
        x=torch.tensor(data_X[sampled_nodes], dtype=torch.float),
        edge_index=torch.tensor(sampled_edges_mapped, dtype=torch.long),
        y=torch.tensor(data_Y[sampled_nodes], dtype=torch.long)
    )

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    train_end = int(data.num_nodes * 0.6)
    val_end = int(data.num_nodes * 0.8)

    data.train_id = np.sort(node_id[:train_end])
    data.val_id = np.sort(node_id[train_end:val_end])
    data.test_id = np.sort(node_id[val_end:])

    if corrected:
        is_mistake = np.loadtxt('pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        mask_train = [i for i in data.train_id if i < len(is_mistake) and not is_mistake[sampled_nodes[i]]]
        mask_val = [i for i in data.val_id if i < len(is_mistake) and not is_mistake[sampled_nodes[i]]]
        mask_test = [i for i in data.test_id if i < len(is_mistake) and not is_mistake[sampled_nodes[i]]]
        data.train_id = np.array(mask_train)
        data.val_id = np.array(mask_val)
        data.test_id = np.array(mask_test)

    data.train_mask = torch.tensor([i in data.train_id for i in range(data.num_nodes)])
    data.val_mask = torch.tensor([i in data.val_id for i in range(data.num_nodes)])
    data.test_mask = torch.tensor([i in data.test_id for i in range(data.num_nodes)])

    return data, data_pubid[sampled_nodes],sampled_nodes


def parse_pubmed():
    path = 'dataset/PubMed_orig/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            label = int(items[1].split('=')[-1]) - \
                1 
            data_Y[i] = label
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy_subgraph(SEED=seed)
    
    if not use_text:
        return data, None

    f = open('dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    path = './responses/spurious/pubmed.json'
    text = []
    rationale = []

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0
    with open(path, 'r',encoding='utf-8') as file:
        json_data = json.load(file)

        from tqdm import tqdm
        for json_example in tqdm(json_data):
            title = json_example['title']
            abstract = json_example['abstract']

            noise = json_example['llm_response']['content_2_round']
                     
            text.append(title+'\n'+abstract+'\n'+noise)
            rationale.append(title+'\n'+abstract)

            cur_text = str(noise+'\n'+title+'\n'+abstract).lower().split()
            cur_rationale = str(title+'\n'+abstract).lower().split()
            if(len(cur_text)>text_max_len):
                text_max_len = len(cur_text)
            if(len(cur_rationale)>rational_max_len):
                rational_max_len = len(cur_rationale)

            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)
    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(text_max_len,rational_max_len,text_tot_len/len(json_data),rational_tot_len/len(json_data),rational_tot_len/text_tot_len))

    return data, text, rationale




def get_raw_text_pubmed_subgraph_aligned(use_text=False, seed=0,use_gpt_explanation=False):
    data, data_pubid,sampled_nodes = get_pubmed_casestudy_subgraph(SEED=seed)
    
    if not use_text:
        return data, None, None

    path = './responses/spurious/pubmed.json'
    text = [""] * len(data_pubid)
    rationale = [""] * len(data_pubid)
    pubid_to_idx = {pubid: idx for idx, pubid in enumerate(data_pubid)}

    text_tot_len = 0.0
    rational_tot_len = 0.0
    text_max_len = 0.0
    rational_max_len = 0.0

    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        for json_example in tqdm(json_data):
            pubid = json_example['pmid']
            if pubid not in pubid_to_idx:
                continue

            idx = pubid_to_idx[pubid]
            title = json_example.get('title', "")
            abstract = json_example.get('abstract', "")
            noise = json_example.get('llm_response', {}).get('content_2_round', "")

            text_entry = noise + '\n' + title + '\n' + abstract
            text_entry = str(title+'\n'+abstract+'\n'+noise)
            rationale_entry = title + '\n' + abstract

            text[idx] = text_entry
            rationale[idx] = rationale_entry

            cur_text = text_entry.lower().split()
            cur_rationale = rationale_entry.lower().split()
            text_max_len = max(text_max_len, len(cur_text))
            rational_max_len = max(rational_max_len, len(cur_rationale))
            text_tot_len += len(cur_text)
            rational_tot_len += len(cur_rationale)

    print("Text MaxLen:{} Rationale MaxLen:{} Text AvgLen:{} Rationale AvgLen:{} Rationale Sparity:{}"
          .format(
              text_max_len,
              rational_max_len,
              text_tot_len / len(text),
              rational_tot_len / len(rationale),
              rational_tot_len / text_tot_len
          ))
    num_edges = data.edge_index.size(1)

    if use_gpt_explanation:
        dataset = 'pubmed'
        folder_path = 'responses/explanation/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]

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


        gpt_explanation = []
        index = 0
        print("sampled_nodes:{}".format(sampled_nodes))
        for node_id in sampled_nodes:

            if(node_id not in fild_ids):
                print("continue: not fild_ids,{}".format(node_id))
                continue
            
            idx = node_id
            content = all_text[node_id]

            gpt_explanation.append(content)
            index = index +1
        
        return data, text, rationale,gpt_explanation


    return data, text, rationale



def get_raw_text_pubmed_generate_data(use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
    if not use_text:
        return data, None

    f = open('dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    PMID = df_pubmed['PMID'].fillna("")
    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text_list = []
    for pmid,ti, ab in zip(PMID,TI, AB):
        text_list.append({
            "pmid":pmid,
            "title":ti,
            "abstract":ab,
        }) 

    return data, text_list