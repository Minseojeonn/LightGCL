import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
from copy import deepcopy
import torchmetrics 
from sklearn.metrics import roc_auc_score


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

# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)
original_train = deepcopy(train)
train = train.tocoo()
print("Nonzero 개수:", train.nnz, train_csr.nnz)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
f = open(path+'valMat.pkl','rb')
val = pickle.load(f)
print('Data loaded.')
print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)
epoch_user = min(train.shape[0], 30000)

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
auc_metric = torchmetrics.AUROC(pos_label = 1)

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

for epoch in range(epoch_no):
    # 모델 저장
    # if (epoch+1)%50 == 0:
        ##torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        ##torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

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
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:  # test every 3 epochs
        val_auc, test_auc = None, None
        with torch.no_grad():
            auc_metric.reset()
            for i, batch in enumerate(tqdm(test_loader)): ## Full batch
                uids, iids, sign = batch
                uids = uids.to(device)
                iids = iids.to(device)
                sign = sign.to(device)
                pred = model(uids, iids, sign, test=True)
                auc_metric.update(pred, sign.long())
            test_auc = auc_metric.compute()
            print(test_auc)
            auc_metric.reset()
            for i, batch in enumerate(tqdm(val_loader)): ## Full batch
                uids, iids, sign = batch
                uids = uids.to(device)
                iids = iids.to(device)
                sign = sign.to(device)
                pred = model(uids, iids, sign, test=True)
                auc_metric.update(pred, sign.long())
            val_auc = auc_metric.compute()
            print(val_auc)
        best_val_auc = max(best_val_auc, val_auc)
        best_test_auc = max(best_test_auc, test_auc)
        print(f"ephoch{epoch} : Best Val AUROC : {best_val_auc}, Best Test AUROC : {best_test_auc}")        

print(f"Result : Best Val AUROC : {best_val_auc}, Best Test AUROC : {best_test_auc}")
