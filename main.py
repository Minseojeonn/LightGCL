import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, load_data, set_random_seed, split_data, txt_to_adj_matrix, logging_with_mlflow, logging_with_mlflow_metric
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
from copy import deepcopy
import torchmetrics 
from sklearn.metrics import roc_auc_score
from utils import metric
import mlflow

device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q
seed = args.seed
use_mlflow = True

# Set MLflow
if use_mlflow:
    remote_server_uri = "http://0.0.0.0:5001"
    mlflow.set_tracking_uri(remote_server_uri)
    experiment_name = f"lightgcl-{args.data}-{args.seed}"
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(vars(args))

# load data
set_random_seed(seed, device)
path = 'data/' + args.data + '.tsv'
##converted_load
edgelist, num_nodes, num_edges = load_data(path, True, "bi")
train_X, train_Y, val_X, val_Y, test_X, test_Y = split_data(edgelist, [0.85, 0.05, 0.1], seed, True)
train = txt_to_adj_matrix(train_X, train_Y, num_nodes)
train_csr = (train!=0).astype(np.float32)
original_train = deepcopy(train)
train = train_csr.tocoo()
test = txt_to_adj_matrix(test_X, test_Y, num_nodes)
val = txt_to_adj_matrix(val_X, val_Y, num_nodes)

# normalizing the adj matrix
rowD = np.array(train_csr.sum(1)).squeeze()
colD = np.array(train_csr.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
    
# construct data loader
train = train.tocoo()
train_data = TrnData(original_train)
train_loader = data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=16)
test = test.tocoo()
test_data = TrnData(test)
test_loader = data.DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=16)
val = val.tocoo()
val_data = TrnData(val)
val_loader = data.DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=16)
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')
auc_metric = torchmetrics.AUROC(task = 'binary')

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')


loss_list = []
loss_r_list = []
loss_s_list = []
best_val_auc, best_test_auc = float("-inf"), float("-inf")



model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr

step_results = []
for epoch in range(epoch_no):
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    for i, batch in enumerate(tqdm(train_loader)): ## batch
        uids, iids, sign = batch
        uids = uids.to(device)
        iids = iids.to(device)
        sign = sign.to(device)
        #feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, sign, test=False)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()
        
    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    
        
    with torch.no_grad():
        test_probs = []
        test_labels = []
        val_probs = []
        val_labels = []
        for i, batch in enumerate(tqdm(test_loader)): ## Full batch
            uids, iids, sign = batch
            uids = uids.to(device)
            iids = iids.to(device)
            sign = sign.to(device)
            prob = model(uids, iids, sign, test=True)
            test_probs.append(prob)
            test_labels.append(sign)    
        for i, batch in enumerate(tqdm(val_loader)): ## Full batch
            uids, iids, sign = batch
            uids = uids.to(device)
            iids = iids.to(device)
            sign = sign.to(device)
            prob = model(uids, iids, sign, test=True)
            val_probs.append(prob)
            val_labels.append(sign) 
        test_probs = torch.cat(test_probs)
        test_labels = torch.cat(test_labels)
        val_probs = torch.cat(val_probs)
        val_labels = torch.cat(val_labels)
        val_metric = metric(val_labels, val_probs)
        test_metric = metric(test_labels, test_probs)        
        step_results.append({"val": val_metric, "test": test_metric})

if use_mlflow:
    logging_with_mlflow_metric(step_results)