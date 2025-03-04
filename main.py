from sched import scheduler

import torch

import numpy as np
from torch.optim.lr_scheduler import StepLR
import random
from data_process import get_bert_emb,get_lightgcl_data,get_interaction
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor,CL_loss
from model import LightGCL
from utils import ModelConfig,TrnData,sampling,get_my_minibatch,MLP
import torch.utils.data as data
from sanfm import SANFM,SANFM_train,top_k
import pickle
from itertools import chain
from tqdm import tqdm
random.seed(8080)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = ModelConfig()
d = config.d
l = config.gnn_layer
temp = config.temp
batch_user = config.batch
epoch_no = config.epoch
lambda_1 = config.lambda1
lambda_2 = config.lambda2
lamdba_3 = config.lamdba3
dropout = config.dropout
lr = config.lr
decay = config.decay
svd_q = config.q
test_epoch = config.test_epoch
test_batch_size = config.test_batch_size
train_batch_size = config.train_batch_size
mashup_nums = config.mashup_nums
api_nums = config.api_nums
k = config.k
sanfm_emb = config.sanfm_emb

def get_Mapping():
    train_mapping = {}
    with open('../dataset/train.txt',"r",encoding='utf-8') as f:
        for lines in f:
            items = lines.strip().split(' ')
            mashup_id = items[0]
            api_list = items[1:]
            api_list = list(map(int, api_list))
            mashup_id = int(mashup_id)
            train_mapping[mashup_id] = api_list
    test_mapping = {}
    with open('../dataset/test.txt', "r", encoding='utf-8') as f:
        for lines in f:
            items = lines.strip().split(' ')
            mashup_id = items[0]
            api_list = items[1:]
            api_list = list(map(int, api_list))
            mashup_id = int(mashup_id)
            test_mapping[mashup_id] = api_list
    # all_mapping = {}
    # with open('../dataset/dataset.txt', "r", encoding='utf-8') as f:
    #     for lines in f:
    #         items = lines.strip().split(' ')
    #         mashup_id = items[0]
    #         api_list = items[1:]
    #         api_list = list(map(int, api_list))
    #         mashup_id = int(mashup_id)
    #         all_mapping[mashup_id] = api_list
    return train_mapping,test_mapping

def main():
    # mapping中记录了mashup和api的交互信息
    long_tail = get_interaction()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    mlp = MLP(768,d,device)
    mlp.to(device)
    train_mapping,test_mapping = get_Mapping()
    mashup_desc_emb, api_desc_emb = get_bert_emb()
    mashup_desc_emb = mlp(mashup_desc_emb)
    mashup_desc_emb = mashup_desc_emb.detach()
    api_desc_emb = mlp(api_desc_emb)
    api_desc_emb = api_desc_emb.detach()
    print(mashup_desc_emb.shape)
    print(api_desc_emb.shape)

    train,test,train_csr,train_edge,test_edge = get_lightgcl_data()

    print(train)

    train_data = TrnData(train)  # 这里使用了一个自定义的类 TrnData 来封装转换后的 train 稀疏矩阵。

    test_data = TrnData(test)
    # print(train_data)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)  # 这段代码调用了一个函数（假设它是你代码中的自定义函数），它的作用是将 train 稀疏矩阵（通常是 scipy.sparse 格式）转换为 PyTorch 的稀疏张量（torch.sparse 格式）。

    adj_norm = adj_norm.coalesce().to(device)

    # adj_norm 是一个张量使用 torch.sparse_coo（稀疏 COO 格式）表示，仅存储非零元素的索引和值。

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)
    print('Performing SVD...')
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
    print("adj =",adj,"q= ",svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    print(svd_u, s, svd_v)
    del s

    print('SVD done.')

    # process test set
    lgcl_test_labels = [[] for i in range(test.shape[0])]  # 这行代码的作用是创建一个嵌套列表，其中包含与 test 数组（或 DataFrame）行数相同的空列表。
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]  # test.data 相当于一个稀疏矩阵然后 （row，col）就相当于非0元素的位置
        lgcl_test_labels[row].append(col)  # test_labels 是一个嵌套列表 然后其中第i个位置就相当于第i个mashup，然后里面的数据就代表着此mashuo所包含的api的编号
    print('Test data processed.')


    model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm,l, temp, lambda_1, lambda_2, lamdba_3,dropout, batch_user, device,input_dim=768,hidden_dim=512,out_dim=d)
    model.to(device)
    sanfm = SANFM(embed_dim=128,att_dim=64,droprate=0.5,i_num=d*2,c_num=0)
    sanfm.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    optimizer = torch.optim.Adam(params=chain(model.parameters(),sanfm.parameters()), lr=lr, weight_decay=0)
    best_recall = 0
    total_recall = 0
    total_ndgc = 0
    total_precision = 0
    test_time = 0
    total_score = 0
    total_map = 0
    for epoch in tqdm(range(1, config.epoch + 1),total=config.epoch):
        model.train()
        sanfm.train()
        uids,pos,neg,mashup_emb,pos_api_emb,neg_api_emb = get_my_minibatch(train_batch_size,train_mapping,api_desc_emb,mashup_desc_emb,train_data,test_data,mashup_nums)

        # print(len(uids), len(pos),len(neg))
        # print(uids.shape, pos.shape, neg.shape,iids.shape)
        optimizer.zero_grad()
        # print(train_mapping)
        loss, loss_r, loss_s,mashup_desc_output,pos_api_desc_output,neg_api_desc_output,mashup_stru_output,api_stru_output = model(uids,  pos, neg,long_tail=long_tail,mapping=train_mapping,pos_api_emb=pos_api_emb,neg_api_emb=neg_api_emb,mashup_emb=mashup_emb)
        mashup_stru_output = mashup_stru_output[uids]
        pos_api_stru_output = api_stru_output[pos]
        neg_api_stru_output = api_stru_output[neg]


        mashup_emb = mashup_desc_output + mashup_stru_output
        pos_api_emb = pos_api_desc_output + pos_api_stru_output
        neg_api_emb = neg_api_desc_output + neg_api_stru_output

        samples,labels = sampling(mashup_emb,pos_api_emb,neg_api_emb)

        sanfm_loss = SANFM_train(sanfm, samples,labels)

        Total_loss = sanfm_loss + loss
        Total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: ",epoch,"Total_loss :",Total_loss.item(),"Lightgcl_loss :",loss.item(),"sanfm_loss :",sanfm_loss.item())
        if epoch % test_epoch == 0:
            test_time +=1
            model.eval()
            sanfm.eval()
            # Lightgcl 测试阶段
            test_uids = np.array([i for i in range(adj_norm.shape[0])])  # test_uids是一个list[0,1.....,2905]
            batch_no = int(np.ceil(len(test_uids) / batch_user))  # ceil:2906/256
            print(batch_no)
            all_recall_20 = 0
            all_ndcg_20 = 0
            for batch in tqdm(range(batch_no)):  # 就是每次取256个 第一次[0,256)  [256,512).....
                start = batch * batch_user
                # print("batch = " ,batch)
                end = min((batch + 1) * batch_user, len(test_uids))
                # print("start = ",start,"end = ",end)
                test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
                # print("test_uids_input = ", test_uids_input )
                predictions = model(test_uids_input, None, None, test=True)
                # print("prediction = ",predictions,"shape = ",predictions.shape)
                predictions = np.array(predictions.cpu())
                # top@10
                recall_20, ndcg_20 = metrics(test_uids[start:end], predictions, k, lgcl_test_labels)
                all_recall_20 += recall_20
                all_ndcg_20 += ndcg_20
            print('-------------------------------------------')
            # sanfm测试阶段
            recall_k, ndcg_k,precision ,MAP= top_k(test_batch_size,model,mashup_desc_emb,api_desc_emb,k,mashup_nums,api_nums,train_mapping,test_mapping,sanfm)
            f1_score = 2*recall_k*precision/(recall_k+precision)
            total_recall+=recall_k
            total_ndgc+=ndcg_k
            total_precision+=precision
            total_score+= f1_score
            total_map += MAP
            if(recall_k>best_recall):
                best_recall = recall_k
            print('Test of epoch', epoch, ':', 'lightgcl_Recall@20:', all_recall_20 / batch_no,"sanfm_Recall@10:",recall_k,"sanfm_ndcg@10:",ndcg_k,"best_test:",best_recall,'precision:',precision)

    print("total_recall:",total_recall/test_time,"total_ndgc:",total_ndgc /test_time,"total_precision:",total_precision/test_time,"total_score:",total_score/test_time,"total_map:",total_map/test_time)
    print(best_recall)
if __name__ == '__main__':
    print(device)
    main()
