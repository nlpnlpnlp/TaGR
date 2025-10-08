import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch.autograd import Function

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class ConvFusion(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_dim*2,  
            out_channels=hidden_dim,    
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, A, B):
        # A, B: [node_num, hidden_dim]
        x = torch.cat([A, B], dim=-1)   # [node_num, 2*hidden_dim]
        x = x.unsqueeze(0)              # [1, node_num, 2*hidden_dim] batch
        x = x.transpose(1, 2)           # [1, 2*hidden_dim, node_num]
        x = self.conv(x)                 # [1, hidden_dim, node_num]
        x = self.bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2).squeeze(0) # [node_num, hidden_dim]
        return x

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


class RationaleDecoder(nn.Module):
    def __init__(self, hidden_dim, max_len, z_dim):
        super(RationaleDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * z_dim)
        )

    def forward(self, x, sampling_fn=None, detach=False):
        z_logits = self.layers(x)  
        z_logits = z_logits.view(*x.shape[:-1], self.max_len, self.z_dim)
        if sampling_fn is not None:  
            z = sampling_fn(z_logits)
        else:
            z = z_logits
        if detach:
            z_logits = z_logits.detach()
            z = z.detach()
        return z, z_logits

class ReLUDropout(nn.Module):
    def __init__(self, dropout: float):
        super(ReLUDropout, self).__init__()
        self.dropout = dropout
    def forward(self, x):
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class LogSoftmax(nn.Module):
    def __init__(self, dim: int = 1):
        super(LogSoftmax, self).__init__()
        self.dim = dim
    def forward(self, x):
        return F.log_softmax(x, dim=self.dim)

class TeacherEncoder(nn.Module):
    def __init__(self, args):
        super(TeacherEncoder, self).__init__()
        self.args = args
        self.z_dim = 2
        self.embedding_layer = Embedding(
            args.vocab_size, args.embedding_dim, args.pretrained_embedding
        )

        self.cls_teacher = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(args.dropout)
        self.graph_backbone = args.graph_backbone  
        self.gnn_layers = nn.ModuleList()

        self.rationale_decoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.args.max_len * self.z_dim)  
        )

        for i in range(args.gnn_layers):
            in_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.hidden_dim
            out_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.num_class
            if self.graph_backbone == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif self.graph_backbone == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            elif self.graph_backbone == "GIN":
                nn_func = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                self.gnn_layers.append(GINConv(nn_func))
            elif self.graph_backbone == "GAT":
                heads = 4 if i < args.gnn_layers - 1 else 1
                self.gnn_layers.append(GATConv(in_dim, out_dim // heads, heads=heads))
            elif self.graph_backbone == "MLP":
                self.gnn_layers.append(nn.Linear(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            
    def _independent_soft_sampling(self, rationale_logits):
        return torch.softmax(rationale_logits, dim=-1)

    def independent_straight_through_sampling(self, rationale_logits):
        return F.gumbel_softmax(rationale_logits, tau=1, hard=True)
    
    def forward(self, inputs, masks, edge_index):
        masks_ = masks.unsqueeze(-1)  
        embedding = masks_ * self.embedding_layer(inputs)  # (batch, seq_len, emb_dim)
        outputs, _ = self.cls_teacher(embedding)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        outputs = torch.transpose(outputs, 1, 2)  
        node_features, _ = torch.max(outputs, dim=2)    # (batch, hidden_dim)

        z_list = []
        x = node_features
        for i, gnn in enumerate(self.gnn_layers):
            if self.graph_backbone in ["GCN", "SAGE", "GIN", "GAT"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone == "MLP":
                x = gnn(x)
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            if i < len(self.gnn_layers)-1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)
        output = F.log_softmax(x, dim=1)
        return output


class StudentTagRnpModel(nn.Module):
    def __init__(self, args):
        super(StudentTagRnpModel, self).__init__()
        self.args = args
        self.z_dim = 2

        self.embedding_layer = Embedding(
            args.vocab_size, args.embedding_dim, args.pretrained_embedding
        )

        self.gen = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)

        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.layernorm_gen = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        
        self.ReLUDropout = ReLUDropout(dropout=self.args.dropout)
        self.LogSoftmax = LogSoftmax(dim=1)

        self.cls = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.cls_fc = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.gcn1 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.gcn2 = GCNConv(args.hidden_dim, args.num_class)

        self.rationale_decoder = RationaleDecoder(
            args.hidden_dim, args.max_len, self.z_dim
        )

        self.graph_backbone = args.graph_backbone  
        self.gnn_layers = nn.ModuleList()

        self.prob_classifier = nn.Linear(args.hidden_dim, args.num_class)

        for i in range(args.gnn_layers):

            in_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.hidden_dim
            out_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.num_class

            if self.graph_backbone == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif self.graph_backbone == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            elif self.graph_backbone == "GIN":
                nn_func = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                self.gnn_layers.append(GINConv(nn_func))
            elif self.graph_backbone == "GAT":
                heads = 4 if i < args.gnn_layers - 1 else 1
                self.gnn_layers.append(GATConv(in_dim, out_dim // heads, heads=heads))
            elif self.graph_backbone == "MLP":
                self.gnn_layers.append(nn.Linear(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")

    def _independent_soft_sampling(self, rationale_logits):
        return torch.softmax(rationale_logits, dim=-1)

    def independent_straight_through_sampling(self, rationale_logits):
        return F.gumbel_softmax(rationale_logits, tau=1, hard=True)

    def forward(self, inputs=None, masks=None, edge_index=None,lm_feature=None,is_evaluating=False):
        masks_ = masks.unsqueeze(-1)  
        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits=self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))
        cls_outputs, _ = self.cls(cls_embedding)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)  
        node_features = self.cls_fc(self.dropout(cls_outputs))  # (batch, hidden_dim)
        z_list =[]
        output_list = []
        x = node_features
        if(lm_feature!=None):
            x = lm_feature
        else:
            x = node_features
        for i, gnn in enumerate(self.gnn_layers):
            if self.graph_backbone in ["GCN", "SAGE"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone in ["GIN", "GAT"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone == "MLP":
                x = gnn(x)
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            if i < len(self.gnn_layers) - 1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

            if(is_evaluating and i< len(self.gnn_layers)-1):
                with torch.no_grad():
                    z_tmp, z_logits = self.rationale_decoder(
                        x,sampling_fn=self.independent_straight_through_sampling,detach=False
                    )
                    z_list.append(z_tmp)

            if(i< len(self.gnn_layers)-1):
                output_tmp = F.log_softmax(self.prob_classifier(x), dim=1)
                output_list.append(output_tmp)
                    
        output = F.log_softmax(x, dim=1)
        if(is_evaluating):
            return z,output, z_list

        return z, output,output_list

class StudentTagRnpModel2(nn.Module):
    def __init__(self, args):
        super(StudentTagRnpModel2, self).__init__()
        self.args = args
        self.z_dim = 2

        self.embedding_layer = Embedding(
            args.vocab_size, args.embedding_dim, args.pretrained_embedding
        )

        self.embedding_layer2 = Embedding(
            args.vocab_size, args.embedding_dim, args.pretrained_embedding
        )

        self.in_channels_dim = args.lm_feature_dim

        self.gen = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)

        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.layernorm_gen = nn.LayerNorm(args.hidden_dim)
        
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        
        self.ReLUDropout = ReLUDropout(dropout=self.args.dropout)
        self.LogSoftmax = LogSoftmax(dim=1)
        self.cls = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.cls2 = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.cls_fc = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.cls_lm_fc = nn.Linear(args.lm_feature_dim, args.hidden_dim)
        self.cls_fc2 = nn.Linear(args.hidden_dim, args.lm_feature_dim)
        self.cls_fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.gcn1 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.gcn2 = GCNConv(args.hidden_dim, args.num_class)

        self.rationale_decoder = RationaleDecoder(
            args.hidden_dim, args.max_len, self.z_dim
        )

        self.graph_backbone = args.graph_backbone  # GCN / SAGE / GIN / RevGAT
        self.gnn_layers = nn.ModuleList()

        self.prob_classifier = nn.Linear(args.hidden_dim, args.num_class)

        for i in range(args.gnn_layers):

            in_dim  = args.hidden_dim if i < args.gnn_layers - 1 else args.hidden_dim
            out_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.num_class


            if self.graph_backbone == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif self.graph_backbone == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            elif self.graph_backbone == "GIN":
                nn_func = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                self.gnn_layers.append(GINConv(nn_func))
            elif self.graph_backbone == "RevGAT":
                heads = 4 if i < args.gnn_layers - 1 else 1
                self.gnn_layers.append(GATConv(in_dim, out_dim // heads, heads=heads))
            elif self.graph_backbone == "MLP":
                self.gnn_layers.append(nn.Linear(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")

    def _independent_soft_sampling(self, rationale_logits):
        return torch.softmax(rationale_logits, dim=-1)

    def independent_straight_through_sampling(self, rationale_logits):
        return F.gumbel_softmax(rationale_logits, tau=1, hard=True)

    def forward(self, inputs=None, masks=None, lm_feature=None,edge_index=None,is_evaluating=False):
        masks_ = masks.unsqueeze(-1)  

        ########## Generator ##########
        embedding = masks_ * self.embedding_layer(inputs) 
        gen_logits=self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))
        cls_outputs, _ = self.cls(cls_embedding)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)  
        psolm_features = self.cls_fc2(self.dropout(cls_outputs))  

        z_list =[]
        output_list = []
        if(lm_feature!=None):  
            node_features = self.cls_lm_fc(lm_feature)  # (batch, hidden_dim)
            x = node_features
        else:
            node_features = self.cls_lm_fc(psolm_features)  # (batch, hidden_dim)
            x = node_features

        for i, gnn in enumerate(self.gnn_layers):
            if self.graph_backbone in ["GCN", "SAGE"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone in ["GIN", "RevGAT"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone == "MLP":
                x = gnn(x)
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            
            if i < len(self.gnn_layers) - 1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

            if(is_evaluating and i< len(self.gnn_layers)-1):
                with torch.no_grad():
                    z_tmp, z_logits = self.rationale_decoder(
                        x,
                        sampling_fn=self.independent_straight_through_sampling,
                        detach=False
                    )
                    z_list.append(z_tmp)

            if(i< len(self.gnn_layers)-1):
                output_tmp = F.log_softmax(self.prob_classifier(x), dim=1)
                output_list.append(output_tmp)
                    
        output = F.log_softmax(x, dim=1)

        if(is_evaluating):
            return z,output, z_list

        return z, output,output_list

    def train_skew(self, inputs=None, masks=None, edge_index=None,skew_masks=None,is_evaluating=False):
        masks_ = masks.unsqueeze(-1)  
        embedding = masks_ * self.embedding_layer2(inputs)  
        cls_embedding = embedding * (skew_masks.unsqueeze(-1))
        cls_outputs, _ = self.cls2(cls_embedding)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)  
        node_features = self.cls_fc3(self.dropout(cls_outputs))  

        z_list =[]
        output_list = []
        x = node_features
        z, z_logits = self.rationale_decoder(
                        node_features,
                        sampling_fn=self.independent_straight_through_sampling,
                        detach=False
                    )

        for i, gnn in enumerate(self.gnn_layers):
            if self.graph_backbone in ["GCN", "SAGE"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone in ["GIN", "RevGAT"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone == "MLP":
                x = gnn(x)
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            
            if i < len(self.gnn_layers) - 1:  
                x = F.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

            if(is_evaluating and i< len(self.gnn_layers)-1):
                with torch.no_grad():
                    z_tmp, z_logits = self.rationale_decoder(
                        x,
                        sampling_fn=self.independent_straight_through_sampling,
                        detach=False
                    )
                    z_list.append(z_tmp)

            if(i< len(self.gnn_layers)-1):
                output_tmp = F.log_softmax(self.prob_classifier(x), dim=1)
                output_list.append(output_tmp)
                    
        output = F.log_softmax(x, dim=1)

        if(is_evaluating):
            return z,output, z_list

        return z, output,output_list


    def get_cls_param(self):
        layers = [self.gnn_layers, self.embedding_layer2,self.cls2,self.cls_fc3]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params

class TagNCModel_BGBBase(nn.Module):
    def __init__(self, args):
        super(TagNCModel_BGBBase, self).__init__()
        self.args = args
        self.z_dim = 2

        self.embedding_layer = Embedding(
            args.vocab_size, args.embedding_dim, args.pretrained_embedding
        )
        self.in_channels_dim = args.ogb_feature_dim
        self.cls = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim // 2,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(args.dropout)
        self.cls_fc = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.graph_backbone = args.graph_backbone  
        self.gnn_layers = nn.ModuleList()

        self.rationale_decoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.args.max_len * self.z_dim)  
        )

        for i in range(args.gnn_layers):
            in_dim = self.in_channels_dim if i == 0 else args.hidden_dim
            out_dim = args.hidden_dim if i < args.gnn_layers - 1 else args.num_class

            if self.graph_backbone == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif self.graph_backbone == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            elif self.graph_backbone == "GIN":
                nn_func = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                self.gnn_layers.append(GINConv(nn_func))
            elif self.graph_backbone == "GAT":
                heads = 4 if i < args.gnn_layers - 1 else 1
                self.gnn_layers.append(GATConv(in_dim, out_dim // heads, heads=heads))
            elif self.graph_backbone == "MLP":
                self.gnn_layers.append(nn.Linear(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")
            
    def _independent_soft_sampling(self, rationale_logits):
        return torch.softmax(rationale_logits, dim=-1)

    def independent_straight_through_sampling(self, rationale_logits):
        return F.gumbel_softmax(rationale_logits, tau=1, hard=True)
    
    def forward(self, x, edge_index,is_evaluating=False):
        
        z_list = []
        for i, gnn in enumerate(self.gnn_layers):
            if self.graph_backbone in ["GCN", "SAGE", "GIN", "GAT"]:
                x = gnn(x, edge_index)
            elif self.graph_backbone == "MLP":
                x = gnn(x)
            else:
                raise ValueError(f"Unknown backbone {self.graph_backbone}")

            if i < len(self.gnn_layers)-1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

            if(is_evaluating and i< len(self.gnn_layers)-1):
                with torch.no_grad():
                    z_logits = self.rationale_decoder(x).detach() 
                    z_logits = z_logits.view(*x.shape[:-1], self.args.max_len, self.z_dim).detach()
                    z_tmp = self.independent_straight_through_sampling(z_logits).detach()
                    z_list.append(z_tmp)
        output = F.log_softmax(x, dim=1)

        if(is_evaluating):
            return z_list,output
        
        return output
