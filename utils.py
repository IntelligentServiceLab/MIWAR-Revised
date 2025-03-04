import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import random
import torch.nn.functional as F
random.seed(8080)
def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])  # 模型为这些用户生成的推荐物品列表，已经按照得分降序排列。
        # print("prediction: ", prediction)
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

class ModelConfig:
    def __init__(self):
        self.gnn_layer = 2
        self.lr = 5e-4 # learning rate
        self.decay =0.99 # learning rate
        self.batch = 256 # batch size
        self.note = None # note
        self.lambda1 = 1e-5 # weight of cl loss
        self.lambda2 = 1e-4 # l2 reg weight
        self.lamdba3 = 1e-3  # LCS weight
        self.epoch = 10000 # number of epochs
        self.d = 64 # embedding size
        self.q = 5 # rank
        self.dropout = 0.5 # rate for edge dropout
        # self.temp=  0.2 # temperature in cl loss
        self.temp = 0.2
        self.seed = 8080
        self.test_epoch = 1000
        # self.cl_rate = 0
        self.test_batch_size = 64
        self.train_batch_size = 64
        self.mashup_nums = 2289
        self.api_nums = 956
        self.k = 5
        self.sanfm_emb = 128
class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

# def my_sampling(train_loader,mapping):
#     uids_list = []
#     pos_list = []
#     neg_list = []
#     l = 0
#     for i,batch in enumerate(train_loader):
#         l +=1
#         uids,pos,neg = batch
#         uids = uids.tolist()
#         pos = pos.tolist()
#         neg = neg.tolist()
#         uids_list.extend(uids)
#         pos_list.extend(pos)
#         neg_list.extend(neg)
#     uids_tensor = torch.tensor(uids_list)
#     pos_tensor = torch.tensor(pos_list)
#     neg_tensor = torch.tensor(neg_list)
#     return uids_tensor, pos_tensor, neg_tensor,l

def sampling(mashup_emb,pos_api_emb,neg_api_emb):

    positive_samples = torch.cat((mashup_emb, pos_api_emb), dim=1)
    negative_samples = torch.cat((mashup_emb, neg_api_emb), dim=1)
    num_positive_samples = positive_samples.shape[0]
    num_negative_samples = negative_samples.shape[0]

    # 创建对应的标签数组
    positive_labels = np.ones((num_positive_samples,))
    negative_labels = np.zeros((num_negative_samples,))
    positive_labels_tensor = torch.tensor(positive_labels, dtype=torch.float32)
    negative_labels_tensor = torch.tensor(negative_labels, dtype=torch.float32)
    samples = torch.cat([positive_samples, negative_samples], dim=0)
    labels = torch.cat([positive_labels_tensor, negative_labels_tensor], dim=0)
    return samples, labels




def CL_loss(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def generate_unique_list(k, x):
    return random.sample(range(0, x), k)


def test_mashup_list(k, x,test_mapping):
    mashup_list = set()
    #生成一个长度为 k，所有数都在 [0, x] 范围内的列表
    for i in range(0,k):
        mashup_id = random.randint(0,x-1)
        while mashup_id in mashup_list or len(test_mapping[mashup_id])<2:
            print(mashup_id, len(test_mapping[mashup_id]))
            mashup_id = random.randint(0, x - 1)
        mashup_list.add(mashup_id)
    # return random.sample(range(0, x), k)
    return mashup_list

class MLP(nn.Module):
    def __init__(self, input_dim,  out_dim, device):
        super(MLP, self).__init__()  # 调用父类的 __init__ 方法
        self.seq = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
        )
        self.device = device
    def forward(self, x):
        x = self.seq(x)
        x.to(self.device)
        return x

def get_my_minibatch(batch_size,train_mapping,api_desc_emb,mashup_desc_emb,data1,data2,mashup_num):
    mashup_list = generate_unique_list(batch_size,mashup_num)
    uids = []
    poss = []
    negs = []
    for mashup_id in mashup_list:
        api_list = train_mapping[mashup_id]
        for api_id in api_list:
            uids.append(mashup_id)
            poss.append(api_id)
            while True:
                neg = random.randint(0,955)
                if (mashup_id, neg) not in data1.dokmat and (mashup_id, neg) not in data2.dokmat:
                    negs.append(neg)
                    break
    mashup_emb = mashup_desc_emb[uids]
    pos_api_emb = api_desc_emb[poss]
    neg_api_emb = api_desc_emb[negs]
    return uids, poss, negs,mashup_emb,pos_api_emb,neg_api_emb