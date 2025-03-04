import torch
import torch.nn as nn
from utils import sparse_dropout, spmm,generate_unique_list
import torch.nn.functional as F
class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, lamdba_3,dropout, batch_user, device,input_dim=None,hidden_dim=None,out_dim=None):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lamdba_3 = lamdba_3
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

        self.relu = nn.ReLU()
        self.api_layers = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, out_dim),
            # nn.Dropout(0.001)
            # nn.Linear(input_dim, out_dim),
            # nn.ReLU(),
            # nn.Dropout(0.01),
            nn.Linear(input_dim,out_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
        )
        self.mashup_layers = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, out_dim),
            # nn.Dropout(0.001)
            # nn.Linear(input_dim, out_dim),
            # nn.ReLU(),
            # nn.Dropout(0.01)
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
        )

    def forward(self, uids, pos, neg, long_tail=None,mapping=None,test=False,pos_api_emb=None,neg_api_emb=None,mashup_emb=None):  #  在默认参数之后不能有非默认参数
        if test==True:  # testing phase
            uids = torch.tensor(uids)
            uids = uids.long().to(self.device)
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray() # 记录了训练集中每个用户对物品的交互关系（1 表示有交互，0 表示无交互）。
            # print("mask = ",mask,"shape = ",mask.shape)
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask  # 这一步将训练集中已交互过的物品的预测得分设置为极小值，以确保这些物品不会出现在推荐列表中。
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            # us = uids
            uids = torch.tensor(uids)
            pos = torch.tensor(pos)
            neg = torch.tensor(neg)
            uids = uids.long().to(self.device)
            pos = pos.long().to(self.device)
            neg = neg.long().to(self.device)
            iids = torch.concat([pos, neg], dim=0)  # 将正样本 pos 和负样本 neg 拼接，生成一个包含所有正负样本的张量
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # extra loss
            # print("mappnig = ",mapping)
            extra_loss = 0
            us = generate_unique_list(64,2289-1)
            for uid in us:
                items = mapping[uid]
                popular =[]
                unpopular =[]
                for item in items:
                    if item in long_tail:
                        unpopular.append(item)
                    else:
                        popular.append(item)
                if len(unpopular) == 0 or len(popular) == 0:
                    continue
                L = 0
                for item1 in unpopular:
                    for item2 in popular:
                        L+=torch.norm(self.E_i[item1] - self.E_i[item2],p=2)
                L = 1/len(popular) * L
                extra_loss += L

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg + self.lamdba_3 * extra_loss
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            pos_api_desc_output = self.process_input(pos_api_emb, "api")

            neg_api_desc_output = self.process_input(neg_api_emb, "api")

            mashup_desc_output = self.process_input(mashup_emb, "mashup")

            return loss, loss_r, self.lambda_1 * loss_s,mashup_desc_output,pos_api_desc_output,neg_api_desc_output,self.E_u,self.E_i

    def process_input(self, emb,type=None):

        if type=="api":
            # processed_output = self.api_layers(emb)
            processed_output = emb
        elif type=="mashup":
            # processed_output=self.mashup_layers(emb)
            processed_output = emb
        return processed_output
