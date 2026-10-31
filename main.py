#!/usr/bin/env python
# coding: utf-8

# In[101]:
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, average_precision_score,precision_recall_curve
import pandas as pd
from sklearn.metrics import roc_curve

from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
early_stopping = EarlyStopping('val_loss', patience=3)
from pytorch_lightning import Trainer
from torch.optim import RAdam
import torch.nn.functional as F
import networkx as nx
import warnings



warnings.filterwarnings("ignore")

import os
from tensorboardX import SummaryWriter


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True




def prepare():

    def PCC(matrix):
        matrix = matrix.corr(method='pearson')
        return matrix
    
    #load the multi-source data
    expression=pd.read_csv('./dataset/scRNA-Seq/STRING_Dataset/mESC/TFs+500/ExpressionData.csv')
    pesudo_time = pd.read_csv('./dataset/Pseudotime/PseudoTime_mESC.csv')
    GO_embedding = np.load("./dataset/GO-hierarchy/GO_GAT_pair_embeddings.npy")
    sequence_embedding = pd.read_csv('./dataset/Sequence/gene_embeddings.csv')
    sequence_embedding.set_index('gene', inplace=True)
    sequence_embedding.index = sequence_embedding.index.astype(str).str.upper()
    
    # expression sorted by pesudo_time
    sort_idx = np.argsort(pesudo_time["PseudoTime"].values)
    ExpressionData_sorted = expression.iloc[sort_idx].reset_index(drop=True)
    ExpressionData_sorted.columns = expression.columns
   
    sim_mat=PCC(expression)
    feature_A = []
    feature_B = []
    feature = []
    traj_feature = []
    label = []

    
    network_TF_target = pd.read_csv('./dataset/scRNA-Seq/STRING_Dataset/mESC/TFs+500/AllPair.csv')
    #network_TF_target = get_low(network_TF_target)
    network_TF_target.columns=['tf','gene','label']
    
    for i in range(len(network_TF_target['label'])):
        tf, gene = network_TF_target.iloc[i, 0], network_TF_target.iloc[i, 1]
        ex_tf, ex_gene = list(sim_mat[tf]), list(sim_mat[gene])
        GO_tf_gene = GO_embedding[i]
        sequence_gene = sequence_embedding.loc[gene].values
        
        ex=np.hstack((ex_tf,ex_gene))
        all_f=np.hstack((ex,GO_tf_gene,sequence_gene))
        
        traj_tf = ExpressionData_sorted[tf].values
        traj_gene = ExpressionData_sorted[gene].values
        traj = np.stack([traj_tf, traj_gene],axis=1)
        traj_feature.append(traj)
       
        feature.append(all_f)
        label.append(network_TF_target.iloc[i, 2])
    
    
    new_feature = np.array(feature)
    traj_feature = np.array(traj_feature)
    new_label = np.array(label)
    event_num = 1
    print("new",new_feature.shape,traj_feature.shape)

    return new_feature,traj_feature,new_label, event_num

# In[105]:


class TGIDataset(Dataset):
    def __init__(self, x,traj_x,y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)
        self.traj_x = torch.FloatTensor(traj_x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index],self.traj_x[index], self.y_data[index]

    def __len__(self):
        return self.len


def add_mask_noise(x, noise_ratio=0.2):
    device = x.device
    batch_size, feature_dim = x.shape
    mask = torch.rand(batch_size, feature_dim, device=device) > noise_ratio
    corrupted_x = x * mask.float()
    return corrupted_x

class DAE(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(DAE, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):

        #if self.training:
            #X = add_mask_noise(X)
        
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att2(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X


#2D Gated Attention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)



class SRMSNorm(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.scale = 1.0 / math.sqrt(emb_dims)
        self.eps = 1e-7

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.norm(x * self.scale, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        return (x / norm).to(x_dtype)


class DropPath(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if not self.training or self.dropout.p == 0.0:
            return x
        B, L, D = x.shape
        mask = torch.ones(B, 1, 1, device=x.device)
        mask = self.dropout(mask)
        return x * mask


class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class LoraBlock(nn.Module):
    def __init__(self, in_dim, out_dim, r):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x))

    def update_weight(self):
        return self.B.weight @ self.A.weight


class MHRetention(nn.Module):
    def __init__(self, emb_dims, num_heads, lth=None, lora=0):
        super().__init__()
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.head_dim = emb_dims // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.lora = lora
        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        self.kernelQ = Kernel()
        self.kernelK = Kernel()
        self.kernelV = nn.Identity()
        self.kernelU = SiLU()

        self.pre_norm = SRMSNorm(emb_dims)
        self.inner_norm = SRMSNorm(self.head_dim)

        if self.lora > 0:
            self.lora_q = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_k = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        if y is None:
            y = x
        B, L1, D = x.shape
        L2 = y.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        u = self.u_proj(x)

        if self.lora > 0:
            q += self.lora_q(x)
            k += self.lora_k(y)
            v += self.lora_v(y)
            u += self.lora_u(x)

        def reshape(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape(q)
        K = reshape(k)
        V = reshape(v)
        U = reshape(u)

        Q = self.kernelQ(Q) / self.scale
        K = self.kernelK(K) / self.scale
        V = self.kernelV(V)
        U = self.kernelU(U)

        if seq_mask is not None:
            Q = Q * seq_mask
        if attn_mask is not None:
            K = K * attn_mask
        if v_pos is not None:
            V = V * v_pos

        score = torch.matmul(Q, K.transpose(-1, -2))          # (B, H, L1, L2)
        self.attn_scores = score.detach()

        KV = torch.matmul(K.transpose(-1, -2), V)
        O = torch.matmul(Q, KV)
        O = self.inner_norm(O) * U

        O = O.transpose(1, 2).contiguous().view(B, L1, D)
        O = self.o_proj(O)
        if self.lora > 0:
            O += self.lora_o(O)
        return O,score


class GatedLinearUnit(nn.Module):
    def __init__(self, emb_dims, lth=None, lora=0):
        super().__init__()
        beta = 1.0 if lth is None else (lth * 8) ** -0.25
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.norm = SRMSNorm(emb_dims)
        self.lora = lora
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)
        if self.lora > 0:
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        u = self.u_proj(x_flat)
        v = self.v_proj(x_flat)

        if self.lora > 0:
            u += self.lora_u(x_flat)
            v += self.lora_v(x_flat)

        out = u * v
        out = self.o_proj(out)

        if self.lora > 0:
            out += self.lora_o(out)

        # out = self.norm(out)
        return out.view(B, L, D)

class RetentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, lth=2, dropout=0.0, lora=0, recompute=False):
        super().__init__()
        self.alpha = (2 * lth) ** 0.25
        self.attn = MHRetention(emb_dims, num_heads, lth, lora)
        self.ffn = GatedLinearUnit(emb_dims, lth, lora)
        self.dropout = nn.Dropout(p=dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)

    def forward(self, x, **kwargs):
        out,score = self.attn(x, **kwargs)
        self.score = score
        out=self.dropout(out)
        x = self.post_norm1(x * self.alpha + out)
        out = self.dropout(self.ffn(x))
        x = self.post_norm2(x * self.alpha + out)
        return x


class CrossRetentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, dropout=0.0, recompute=False):
        super().__init__()
        self.attn1 = MHRetention(emb_dims, num_heads)
        self.attn2 = MHRetention(emb_dims, num_heads)
        self.ffn = GatedLinearUnit(emb_dims)
        self.dropout = nn.Dropout(p=dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)
        self.post_norm3 = nn.LayerNorm(emb_dims)

    def forward(self, x, y, v_pos=None, attn_mask=None, seq_mask=None):
        out = self.attn1(x)
        x = self.post_norm1(x + self.dropout(out))
        out = self.attn2(x, y=y, v_pos=v_pos, attn_mask=attn_mask, seq_mask=seq_mask)
        x = self.post_norm2(x + self.dropout(out))
        out = self.ffn(x)
        x = self.post_norm3(x + self.dropout(out))
        return x


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BiGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=64,
            num_layers=1,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            128,
            812
        )

    def forward(self,x):

        out,_ = self.gru(x)

        out = out[:,-1,:]
        #out = torch.mean(out, dim=1)

        out = self.fc(out)

        return out


class BERT(torch.nn.Module):
    def __init__(self,input_dim,n_heads,n_layers,event_num):
        super(BERT, self).__init__()

        self.expr_dim = input_dim - 256 - 768
        
        self.gru = BiGRU()
        self.dae = DAE(self.expr_dim)
        self.proj_GO = torch.nn.Linear(256,812)
        self.proj_sequence = torch.nn.Linear(768,812)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim=input_dim
        
        self.layers = torch.nn.ModuleList([RetentionLayer(812,n_heads) for _ in range(n_layers)])
        self.AN=torch.nn.LayerNorm(812)
        
        self.l1=torch.nn.Linear(812,(len_after_AE+event_num)//2)
        self.bn1=torch.nn.BatchNorm1d((len_after_AE+event_num)//2)
        
        self.l2=torch.nn.Linear((len_after_AE+event_num)//2,event_num)
        
        self.ac=gelu

    def forward(self, X,traj_x):

        X_expr = X[:, :self.expr_dim]                                # 取前 expr_dim 个
        X_GO = X[:, self.expr_dim : self.expr_dim + 256]             # 取中间 256 个
        X_sequence = X[:, self.expr_dim + 256:]                      # 取最后 768 个

        embedding_expr, X_AE = self.dae(X_expr)
        embedding_GO = self.proj_GO(X_GO)
        embedding_sequence = self.proj_sequence(X_sequence)
        X_traj = self.gru(traj_x)

        X_expr_3d = embedding_expr.unsqueeze(1)       # (B, 1, 812)
        X_GO_3d = embedding_GO.unsqueeze(1)           # (B, 1, 812)
        X_seq_3d = embedding_sequence.unsqueeze(1)    # (B, 1, 812)
        X_traj_3d = X_traj.unsqueeze(1)               # (B, 1, 812)


        X = torch.cat([X_expr_3d,X_GO_3d,X_seq_3d,X_traj_3d], dim=1)
        

        for layer in self.layers:
            X = layer(X)

        X = X.mean(dim=1)  
        X = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        return X,X_AE



class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class my_loss(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.BCEWithLogitsLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs):
        loss = calssific_loss_weight * self.criteria1(X, target.float()) 
        return loss


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.BCEWithLogitsLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE):
        expr_inputs = inputs[:, :X_AE.size(1)] 
        loss = calssific_loss_weight * self.criteria1(X, target.float()) + \
               self.criteria2(expr_inputs.float(), X_AE) 
        return loss


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


# In[115]:
def BERT_train(model, x_train,traj_train, y_train, x_test, traj_test , y_test, event_num):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    '''
    indices = np.random.choice(len(x_train), int(len(x_train)//1.65), replace=False)
    print("indices",indices)
    print("Train len:",indices.shape)
    np.save('./potential_GRN/train_index_mDC.npy', indices)
    x_train, y_train = x_train[indices], y_train[indices]
    '''

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2]))))
    traj_train = np.vstack((traj_train,traj_train))
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(traj_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    
    len_train = len(y_train)
    len_test = len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = TGIDataset(x_train,traj_train, np.array(y_train))
    test_dataset = TGIDataset(x_test,traj_test, np.array(y_test))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epo_num):
        if epoch < epoch_changeloss:
            my_loss = my_loss1()
        else:
            my_loss = my_loss1()

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x,traj_x,y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).to(x.device)  # 确保 index 在与 x 相同的设备上
            
            inputs = lam * x + (1 - lam) * x[index, :]
            traj_mix = (lam * traj_x + (1-lam) * traj_x[index])
            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            traj_mix = traj_mix.to(device)
            targets_a = targets_a.to(device).unsqueeze(1)
            targets_b = targets_b.to(device).unsqueeze(1)

            model_optimizer.zero_grad()
            X,X_AE = model(inputs.float(),traj_mix.float())
            loss = lam * my_loss(X, targets_a, inputs,X_AE) 

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        pre_score = np.zeros((0, event_num), dtype=float)
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader, 0):
                inputs,inputs_traj,target = data

                inputs = inputs.to(device)
                inputs_traj = inputs_traj.to(device)

                target = target.to(device).unsqueeze(1)

                X,X_AE = model(inputs.float(),inputs_traj.float())
                loss = my_loss(X, target, inputs,X_AE)
                testing_loss += loss.item()
                pre_score = np.vstack((pre_score, X.cpu().numpy()))

            
            result_,pred_one=(cal_metrics(y_test,F.sigmoid(torch.Tensor(pre_score))))
            print(result_)
            #np.save("./AUROC/pre_socre"+str(epoch)+".npy",F.sigmoid(torch.Tensor(pre_score)))
            #np.save("./AUROC/y_test.npy",y_test)
        print('epoch [%d] loss: %.6f testing_loss: %.6f ' % (
        epoch + 1, running_loss / len_train, testing_loss / len_test))

    
    torch.save(model.state_dict(), "model_weights.pth")
    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs,inputs_traj, _ = data
            inputs = inputs.to(device)
            inputs_traj = inputs_traj.to(device)
            X,X_AE = model(inputs.float(),inputs_traj.float())
            pre_score = np.vstack((pre_score,X.cpu().numpy()))
    return pre_score


def cal_metrics(label,pred):

    def Find_Optimal_Cutoff(TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold

    pred_one=[]
    fpr, tpr, thresholds = metrics.roc_curve(label,pred, pos_label=1)
    thre=Find_Optimal_Cutoff(tpr, fpr, thresholds)

    for i in pred:
        if i>thre:
            pred_one.append(1)
        else:
            pred_one.append(0)

    auc=metrics.auc(fpr, tpr)
    acc=accuracy_score(label,pred_one)
    recall=recall_score(label,pred_one)
    precision=precision_score(label,pred_one)
    f1=f1_score(label,pred_one)
    precision_, recall_, thresholds_ = precision_recall_curve(label,pred_one)
    aupr=metrics.auc(recall_, precision_)
    return (auc,acc,recall,precision,f1,aupr),pred_one

# In[117]:
def cross_val(feature,traj_feature, label, event_num):
    global test_pairs_all
    skf = StratifiedKFold(n_splits=cross_ver_tim)
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    results=[]
    result=np.zeros(6)
    od=0
    for train_index, test_index in skf.split(feature, label):
       
        
        model = BERT(len(feature[0]), bert_n_heads, bert_n_layers, event_num)
        print("len(feature[0])",len(feature[0]))
       
        X_train, X_test = feature[train_index], feature[test_index]
        traj_train, traj_test = (traj_feature[train_index],traj_feature[test_index])
        y_train, y_test = label[train_index], label[test_index]
        
        pred_score=BERT_train(model, X_train,traj_train, y_train, X_test,traj_test, y_test, event_num)

        result_,pred_one=(cal_metrics(y_test,F.sigmoid(torch.Tensor(pred_score))))
        result_=np.array(result_)
        print(result_)
        print("saved")
        result += np.array(result_)
    result=(result/5)
    print('AUC:',result[0])
    print('ACC:',result[1])
    print('Recall:',result[2])
    print('Precision:',result[3])
    print('F1:',result[4])
    print('AUPR:',result[5])
    print('y_true',y_true)
# In[118]:


file_path="/home/zhongle/Data/"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=2
drop_out_rating=0.3
batch_size=128
len_after_AE= 812
learn_rating=0.00001
epo_num=60
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss = epo_num // 2
weight_decay_rate=0.0001
test_pairs_all=pd.DataFrame()

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task1'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


# In[119]:


def main():
    
    new_feature,traj_feature, new_label, event_num=prepare()
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    print("dataset len", len(new_feature))
    
    start=time.time()
    cross_val(new_feature,traj_feature, new_label,event_num)
    #del test_pairs_all['score']
    #test_pairs_all.to_csv('~/test_R/test_venn/mDC-1000.csv',index=False)
    #result_all, result_eve=cross_val(new_feature,new_label,event_num,tf_gene)
    print("time used:", (time.time() - start) / 3600)
    #save_result(file_path,"all",result_all)
    #save_result(file_path,"each",result_eve)


# In[120]:


main()

