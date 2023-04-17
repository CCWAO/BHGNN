import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, ARMAConv, ChebConv
from sparse_softmax import Sparsemax
from torch_geometric.utils import softmax
from utils import hp_structure
from graphmodel import GCNmodel
from aram_bayes import ARMAConv_Bayes
from chebconv_bayes import ChebConv_bayes


class un_loss(nn.Module):
    def __init__(self, config, size_average=True):
        super(un_loss, self).__init__()

        self.size_average = size_average
        self.device = config.device

    def forward(self, prediction, label, output, time_output, un_weight):
        
        un_weight = un_weight / un_weight.max()
        un_weight = un_weight.gather(1, label.view(-1,1))
        un_weight = un_weight.to(self.device)

        loss_ce = F.nll_loss(F.log_softmax(prediction, dim=1), label, reduction='none')
        loss_mse = F.mse_loss(output, time_output, reduction='none')

        loss_mse_un = torch.mul(torch.exp(un_weight), loss_mse)

        if self.size_average:
            loss_ce = loss_ce.mean()
            loss_mse_un = loss_mse_un.mean()
        else:
            loss_ce = loss_ce.sum()
            loss_mse_un = loss_mse_un.sum()

        return loss_ce + loss_mse_un
        

class Edgelayer(nn.Module):
    def __init__(self, channels, sparse=True, negative_slop=0.2):
        super(Edgelayer, self).__init__()
        self.channels = channels
        self.negative_slop = negative_slop
        self.sparse = sparse

        self.att = Parameter(torch.Tensor(1, self.channels * 2))

        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()

    def forward(self, x, edge_index):

        row, col = edge_index  # inital row col
        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)

        weights = F.leaky_relu(weights, self.negative_slop)
        if self.sparse:
            new_edge_attr = self.sparse_attention(weights, row)
        else:
            new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
        ind = torch.where(new_edge_attr != 0)[0]
        new_edge_index = edge_index[:, ind]
        new_edge_attr = new_edge_attr[ind]

        return x, new_edge_index, new_edge_attr

class EModel_block(torch.nn.Module):
    def __init__(self, config):
        super(EModel_block, self).__init__()
        self.num_features_inner = config.num_features_inner
        self.nhid = config.nhid
        self.dataset_name = config.data_name
        self.device = config.device
        self.dropout_rate = 0.05
        self.layer_name = config.layer_name

        self.edge_inner = Edgelayer(self.num_features_inner, sparse=True, negative_slop=0)
        self.edge_outer = Edgelayer(self.nhid * 2, sparse=False, negative_slop=0.2)

        if self.layer_name == 'GCN':
            self.conv_inner = GCNConv(self.num_features_inner, self.nhid)
            self.conv_outer_1 = GCNmodel(self.nhid * 2, self.nhid, dropout=self.dropout_rate)
            self.conv_outer_2 = GCNmodel(self.nhid, self.nhid, dropout=self.dropout_rate)
        elif self.layer_name == 'aram':
            self.conv_inner = ARMAConv(self.num_features_inner, self.nhid)
            self.conv_outer_1 = ARMAConv_Bayes(self.nhid * 2, self.nhid, dropout=self.dropout_rate)
            self.conv_outer_2 = ARMAConv_Bayes(self.nhid, self.nhid, dropout=self.dropout_rate)
        elif self.layer_name == 'che':
            self.conv_inner = ChebConv(self.num_features_inner, self.nhid, K=3)
            self.conv_outer_1 = ChebConv_bayes(self.nhid * 2, self.nhid, K=3, dropout=self.dropout_rate)
            self.conv_outer_2 = ChebConv_bayes(self.nhid, self.nhid, K=3, dropout=self.dropout_rate)
            
    def forward(self, x, edge_index, batch):
        x, new_edge_index, new_edge_attr = self.edge_inner(x, edge_index)
        x = F.relu(self.conv_inner(x, new_edge_index, new_edge_attr))
        x_hypernode, edge_hypernode, batch_hyper = self.Nodefeature_pool(x, batch)
        x_hyper, edge_index_hyper, edge_attr_hyper = self.edge_outer(x_hypernode, edge_hypernode)
        x_hyper = F.relu(self.conv_outer_1(x_hyper, edge_index_hyper, edge_attr_hyper))
        x_hyper = F.relu(self.conv_outer_2(x_hyper, edge_index_hyper, edge_attr_hyper))

        return x_hyper, batch_hyper

    def Nodefeature_pool(self, x, batch):
        edge_hypernode, batch_inner = hp_structure(self.dataset_name)
        batch_inner = batch_inner.to(self.device)
        batch_hp = []
        x_hp = []
        for i in range(batch.max() + 1):
            ind = batch == i
            x_tem = x[ind, :]
            ht_tem = torch.cat([global_mean_pool(x_tem, batch_inner), global_max_pool(x_tem, batch_inner)], dim=1)
            x_hp.append(ht_tem)
            batch_hp.append(i * torch.ones((6,), dtype=torch.int64))
        batch_hypernode = torch.squeeze(torch.cat(batch_hp, dim=0))
        x_hypernode = torch.squeeze(torch.cat(x_hp, dim=0))

        return x_hypernode, edge_hypernode.to(self.device), batch_hypernode.to(self.device)


class BHGNN(torch.nn.Module):
    def __init__(self, config):
        super(BHGNN, self).__init__()

        self.nhid = config.nhid
        self.num_classes = config.num_classes
        self.dropout_rate = config.drop_rate
        self.test_loop = config.n_samples
        self.num_layer = config.num_layers


        self.graph_block1 = EModel_block(config)

        self.fc1_mean = nn.Linear(self.nhid * 2, self.nhid * 2)
        self.fc2_mean = nn.Linear(self.nhid * 2, self.nhid)

        self.fc1_var = nn.Linear(self.nhid * 2, self.nhid * 2)
        self.fc2_var = nn.Linear(self.nhid * 2, self.nhid)

        self.classifier_mean = nn.Linear(self.nhid, self.num_classes)
        self.classifier_var = nn.Linear(self.nhid, self.num_classes)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        z1, batch_hyper = self.graph_block1(x, edge_index, batch)
        x1 = torch.cat([global_mean_pool(z1, batch_hyper), global_max_pool(z1, batch_hyper)], dim=1)

        x1_mean = F.relu(F.dropout(self.fc1_mean(x1), p=self.dropout_rate))
        x1_mean = F.relu(F.dropout(self.fc2_mean(x1_mean), p=self.dropout_rate))

        x1_var = F.relu(F.dropout(self.fc1_var(x1), p=self.dropout_rate))
        x1_var = F.relu(F.dropout(self.fc2_var(x1_var), p=self.dropout_rate))

        x_classify_mean = self.classifier_mean(x1_mean)
        x_classify_var = self.classifier_var(x1_var)

        return x_classify_mean, x_classify_var, x1_mean

    def test_un(self, data):
    
        mean_class_tem = []
        var_ep_class_tem = []
        var_ae_class_tem = []
        
        mean_out_tem = []
        
        for i_sample in range(self.test_loop):
            x_classify_mean, x_classify_var, _ = self.forward(data)
            mean_out_tem.append(x_classify_mean)
            mean_class_tem.append(F.softmax(x_classify_mean, dim=-1))
            var_ep_class_tem.append(F.softmax(x_classify_mean, dim=-1) ** 2)
            var_ae_class_tem.append(F.softplus(x_classify_var))
        
        mean_out = torch.stack(mean_out_tem, dim=0).mean(dim=0)
        mean_class = torch.stack(mean_class_tem, dim=0).mean(dim=0)  # x mean
        var_ep_class = torch.stack(var_ep_class_tem, dim=0).mean(dim=0)  # x**2 mean
        
        ep = var_ep_class - mean_class ** 2
        ae = torch.stack(var_ae_class_tem, dim=0).mean(dim=0)
        var_class = ep + ae


        results = {'sample_id': data.time_id, 'mean': mean_out, 'ep': ep, 'ae': ae, 'un': var_class}

        return results
        

