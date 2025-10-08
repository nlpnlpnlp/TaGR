## Download TAG datasets and LLM responses from TAPE 

You need to follow He et al. [1] to obtain the original text attributes and LLM-generated responses from TAPE. All of these data are publicly available.

### A. Original text attributes

| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The OGB provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset, unzip and move it to `dataset/ogbn_arxiv_orig`.|
| ogbn-products |  The dataset is located under `dataset/ogbn_products_orig`.|
| arxiv_2023 |  Download the dataset, unzip and move it to `dataset/arxiv_2023_orig`.|
|Cora| Download the dataset, unzip and move it to `dataset/cora_orig`.|
PubMed | Download the dataset, unzip and move it to `dataset/PubMed_orig`.|


### B. LLM responses
| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | Download the dataset, unzip and move it to `gpt_responses/ogbn-arxiv`.|
| ogbn-products  | Download the dataset, unzip and move it to `gpt_responses/ogbn-products`.|
| arxiv_2023 | Download the dataset, unzip and move it to `gpt_responses/arxiv_2023`.|
|Cora| Download the dataset, unzip and move it to `gpt_responses/cora`.|
PubMed | Download the dataset, unzip and move it to `gpt_responses/PubMed`.|

### Reference 
[1] X. He, X. Bresson, T. Laurent, A. Perold, Y. LeCun, and B. Hooi. Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning. In The Twelfth International Conference on Learning Representations. 2024. 
