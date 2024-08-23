import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.dokmat, self.train_positive_rows, self.train_positive_cols, self.train_negative_rows, self.train_negative_cols, self.train_negative, self.train_positive = self.positive_negative_edge_spliter(coomat)
        self.train_rows = torch.tensor(coomat.row, device="cpu", dtype=torch.int64) #long
        self.train_cols = torch.tensor(coomat.col, device="cpu", dtype=torch.int64) #long
        self.dokmat = torch.tensor(self.dokmat.toarray(), device="cpu", dtype=torch.float32) #float

    def positive_negative_edge_spliter(self, coomat):
        train_positive = (coomat==1).astype(np.float32).tocoo() #1만 1로, -1은 0으로 매핑 하니까, sign문제 해결.
        train_negative = (coomat==-1).astype(np.float32).tocoo()  
        return train_positive.todok(), train_positive.row, train_positive.col, train_negative.row, train_negative.col, train_negative.tocsr(), train_positive.tocsr()
  
    def __len__(self):
        return len(self.train_rows)

    def __getitem__(self, idx):
        return self.train_rows[idx], self.train_cols[idx], self.dokmat[self.train_rows[idx],self.train_cols[idx]]
    
    
def load_data(
    dataset_path: str,
    direction: bool,
    node_idx_type: str
) -> np.array:
    """Read data from a file

    Args:
        dataset_path (str): dataset_path
        direction (bool): True=direct, False=undirect
        node_idx_type (str): "uni" - no intersection with [uid, iid], "bi" - [uid, iid] idx has intersection

    Return:
        array_of_edges (array): np.array of edges
        num_of_nodes: [type1(int), type2(int)]
    """
    edgelist = []
    with open(dataset_path) as f:
        for line in f:
            a, b, s = map(int, line.split('\t'))
            if s == -1:
                s = -1
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)

    if node_idx_type.lower() == "uni":
        for idx, edge in enumerate(edgelist.tolist()):
            fr, to, sign = edge
            edgelist[idx] = (fr, to+num_of_nodes[0], sign)
        edgelist = np.array(edgelist)
        assert len(set(edgelist[:, 0].tolist()).intersection(
            set(edgelist[:, 1].tolist()))) == 0, "something worng"

    if direction == False:
        edgelist = edgelist.tolist()
        for idx, edgelist in enumerate(edgelist):
            fr, to, sign = edgelist
            edgelist.append(to, fr, sign)
        edgelist = np.array(edgelist)

    num_edges = np.array(edgelist).shape[0]

    if node_idx_type.lower() == "bi" and direction == False:
        raise Exception("undirect can not use with bi type.")

    return edgelist, num_of_nodes, num_edges

def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
def split_data(
    array_of_edges: np.array,
    split_ratio: list,
    seed: int,
    dataset_shuffle: bool,
) -> dict:
    """Split your dataset into train, valid, and test

    Args:
        array_of_edges (np.array): array_of_edges
        split_ratio (list): train:test:val = [float, float, float], train+test+val = 1.0 
        seed (int) = seed
        dataset_shuffle (bool) = shuffle dataset when split

    Returns:
        dataset_dict: {train_edges : np.array, train_label : np.array, test_edges: np.array, test_labels: np.array, valid_edges: np.array, valid_labels: np.array}
    """

    assert np.isclose(sum(split_ratio), 1), "train+test+valid != 1"
    train_ratio, valid_ratio, test_ratio = split_ratio
    train_X, test_val_X, train_Y, test_val_Y = train_test_split(
        array_of_edges[:, :2], array_of_edges[:, 2], test_size=1 - train_ratio, random_state=seed, shuffle=dataset_shuffle)
    val_X, test_X, val_Y, test_Y = train_test_split(test_val_X, test_val_Y, test_size=test_ratio/(
        test_ratio + valid_ratio), random_state=seed, shuffle=dataset_shuffle)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def txt_to_adj_matrix(edgelist, edgesign, num_nodes):
    adj_matrix = np.zeros(num_nodes, dtype=int)
    for edge, sign in zip(edgelist, edgesign):
        i, j = edge
        adj_matrix[i, j] = sign  # directed_graph, if undirected_graph, alredy augmented. and no sign.
    adj_matrix = coo_matrix(adj_matrix)
    return adj_matrix

def get_num_nodes(
    dataset: np.array
) -> int:
    """get num nodes when bipartite

    Args:
        dataset (np.array): dataset

    Returns:
        num_nodes tuple(int, int): num_nodes_user, num_nodes_item
    """
    num_nodes_user = np.amax(dataset[:, 0]) + 1
    num_nodes_item = np.amax(dataset[:, 1]) + 1
    return (num_nodes_user.item(), num_nodes_item.item())


def logging_with_mlflow(epochs, val_metric, test_metric):
    for idx in tqdm(range(epochs), desc='mlflow_uploading'):
        mlflow.log_metric("val_auc", val_metric["auc"][idx], step=idx)
        mlflow.log_metric(
            "val_binary_f1", val_metric["f1-binary"][idx], step=idx)
        mlflow.log_metric(
            "val_macro_f1", val_metric["f1-macro"][idx], step=idx)
        mlflow.log_metric(
            "val_micro_f1", val_metric["f1-macro"][idx], step=idx)
        mlflow.log_metric("test_auc", test_metric["auc"][idx], step=idx)
        mlflow.log_metric("test_binary_f1",
                          test_metric["f1-binary"][idx], step=idx)
        mlflow.log_metric(
            "test_macro_f1", test_metric["f1-macro"][idx], step=idx)
        mlflow.log_metric(
            "test_micro_f1", test_metric["f1-micro"][idx], step=idx)
        

def logging_with_mlflow_metric(results):
    # metric selected by valid score
    best_auc_epoch, best_auc_score = -1, -float("inf")
    best_macro_epoch, best_macro_score = -1, -float("inf")
    best_binary_epoch, best_binary_score = -1, -float("inf")
    best_micro_epoch, best_micro_score = -1, -float("inf")
    for idx in tqdm(range(len(results)), desc='mlflow_uploading'):
        train_metric, val_metric, test_metric, train_loss = [
            value for key, value in results[idx].items()]
        val_auc, val_bi, val_mi, val_ma = [v for k, v in val_metric.items()]
    
        if best_auc_score <= val_auc:
            best_auc_score = val_auc
            best_auc_epoch = idx
        if best_binary_score <= val_bi:
            best_binary_epoch = val_bi
            best_binary_epoch = idx
        if best_micro_score <= val_mi:
            best_micro_score = val_mi
            best_micro_epoch = idx
        if best_macro_score <= val_ma:
            best_macro_epoch = val_ma
            best_macro_epoch = idx
    
        metrics_dict = {
            "train_auc": train_metric["auc"],
            "train_binary_f1": train_metric["f1-bi"],
            "train_macro_f1": train_metric["f1-ma"],
            "train_micro_f1": train_metric["f1-mi"],
            "val_auc": val_metric["auc"],
            "val_binary_f1": val_metric["f1-bi"],
            "val_macro_f1": val_metric["f1-ma"],
            "val_micro_f1": val_metric["f1-mi"],
            "test_auc": test_metric["auc"],
            "test_binary_f1": test_metric["f1-bi"],
            "test_macro_f1": test_metric["f1-ma"],
            "test_micro_f1": test_metric["f1-mi"],
            "loss_sum": train_loss["loss_sum"],
            "sign_loss": train_loss["sign_loss"]
        }
        if train_loss["model_loss"] != None:
            metrics_dict["cl_loss"] = train_loss["model_loss"]
        mlflow.log_metrics(metrics_dict, synchronous=False, step=idx)

    best_metrics_dict = {
        "best_auc_val": results[best_auc_epoch]["valid"]["auc"],
        "best_bi_val": results[best_binary_epoch]["valid"]["f1-bi"],
        "best_ma_val": results[best_macro_epoch]["valid"]["f1-ma"],
        "best_mi_val": results[best_micro_epoch]["valid"]["f1-mi"],
        "best_auc_test": results[best_auc_epoch]["test"]["auc"],
        "best_bi_test": results[best_binary_epoch]["test"]["f1-bi"],
        "best_ma_test": results[best_macro_epoch]["test"]["f1-ma"],
        "best_mi_test": results[best_micro_epoch]["test"]["f1-mi"]
    }
    mlflow.log_metrics(best_metrics_dict, synchronous=True)