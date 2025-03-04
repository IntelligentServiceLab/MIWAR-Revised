# 导入所需的库
import pickle
from distutils.command.config import config

import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import numpy as np
import time
from tqdm import tqdm
import os
import glob
import random
# from MLP_MAIN import MLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(8080)
# 从Hugging Face模型库中加载预训练的BERT模型和分词器
model = BertModel.from_pretrained('./bert')
model.to(device)
tokenizer = BertTokenizer.from_pretrained('./bert')

# 检查文件路径是否正确
def check_for_pt_files(file_name,folder_path):
    # 使用 glob 模块匹配文件路径模式
    pt_files = glob.glob(os.path.join(folder_path, f'{file_name}.pt'))
    # 检查是否有匹配的文件
    if pt_files:
      return True
    else:
      return False

def get_bert_emb():
    # flag1 和 flag2 判断之前是否生成过mashup和api的嵌入表示
    flag1 = check_for_pt_files('mashup_descr_emb',"../dataset")
    flag2 = check_for_pt_files('api_descr_emb',"../dataset")
    print(flag1,flag2)
    # 从CSV文件中加载数据
    mashup_data = pd.read_csv("../dataset/Mashup_desc.csv", encoding='UTF-8', header=0)  # 使用Mashups.csv文件
    api_data = pd.read_csv("../dataset/API_desc.csv", encoding='UTF-8', header=0)  # 使用APIs.csv文件

    # 从数据中提取描述信息列
    mashup_descr = mashup_data['description']
    api_descr = api_data['description']

    # 打印描述数据的形状（行数，列数）
    print("shape of mashup_desc ", mashup_descr.shape)
    print("shape of api_desc ", api_descr.shape)

    mashup_descr_emb = bert_convert_emb(mashup_descr) if flag1==False else torch.load('../dataset/mashup_descr_emb.pt',map_location=device)

    api_descr_emb = bert_convert_emb(api_descr) if flag2==False else torch.load('../dataset/api_descr_emb.pt',map_location=device)
    if not flag1:
        torch.save(mashup_descr_emb,'../dataset/mashup_descr_emb.pt')
    if not flag2:
        torch.save(api_descr_emb,'../dataset/api_descr_emb.pt')
    return mashup_descr_emb, api_descr_emb

def encode_data(descriptions):
    input_ids = []
    attention_masks = []
    for desc in descriptions:
        encoded_desc = tokenizer.encode_plus(desc, add_special_tokens=True, max_length=150,
                                             padding='max_length',
                                             truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_desc['input_ids'])
        attention_masks.append(encoded_desc['attention_mask'])
    #这两行代码的作用是将多个张量（input_ids 和 attention_masks）沿指定维度（这里是 dim=0）进行拼接，生成一个更大的张量，通常用于将多个样本的输入或掩码合并为一个批次（batch），以便批量处理。
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids.to(device), attention_masks.to(device)  # 将数据移动到 CUDA 设备上

def bert_convert_emb(descriptions):

    input_ids, attention_masks = encode_data(descriptions)

    # 将 input_ids 和 attention_masks 封装成 TensorDataset 将多个张量（例如 input_ids 和 attention_masks）组合在一起，形成一个统一的数据集。
    dataset = TensorDataset(input_ids, attention_masks)

    # 定义批次大小
    batch_size = 512

    # 创建 DataLoader 对象，用于分批次加载数据
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # 获取总批次数量
    total_batches = len(data_loader)
    print("total batches: ", total_batches)
    # 遍历每个批次，并在每个批次上进行模型推理
    all_sentence_vectors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing", total=total_batches)):
            batch_input_ids, batch_attention_masks = batch
            # 将 input_ids 和 attention_masks 传入 BERT 模型
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            # 获取 BERT 模型的文本向量表示（最后一层的隐藏状态）
            batch_sentence_vectors = outputs[1] # 获取 [CLS] token 的隐藏状态作为句子向量
            all_sentence_vectors.append(batch_sentence_vectors)
            # 更新进度条

    # 将每个批次的文本向量拼接起来
    all_sentence_vectors = torch.cat(all_sentence_vectors, dim=0)

    # all_sentence_vectors 就是每个描述文本的 BERT 表示（文本向量）

    return all_sentence_vectors

def get_lightgcl_data():
    path = '../dataset/'
    f = open(path + 'trnMat.pkl', 'rb')
    train = pickle.load(f)
    # train为一个COO结构也就是一个 SciPy 的稀疏矩阵， (x,y)  k   (行索引，列索引)  值
    # print("train :"+"\n",train)
    train_csr = (train != 0).astype(
        np.float32)  # 这段代码的作用是将一个稀疏矩阵 train 中的非零元素转换为 1.0，并将零元素转换为 0.0，然后将其转换为 np.float32 类型。
    f = open(path + 'tstMat.pkl', 'rb')
    test = pickle.load(f)
    epoch_user = min(train.shape[0], 30000)
    # normalizing the adj matrix 归一化
    rowD = np.array(train.sum(1)).squeeze()  # 计算的是每一行的非零元素的和。
    colD = np.array(train.sum(0)).squeeze()  # 计算的是每一列的非零元素的和。
    train_interaction_edge = [[], []]
    test_interaction_edge = [[], []]
    for i in range(len(train.data)):
        train_interaction_edge[0].append(train.row[i])
        train_interaction_edge[1].append(train.col[i])
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)
    for i in range(len(test.data)):
        test_interaction_edge[0].append(test.row[i])
        test_interaction_edge[1].append(test.col[i])
    train = train.tocoo()  # train.tocoo()：这是将 train 稀疏矩阵转换为 COO 格式（坐标格式）。
    print(train_interaction_edge, test_interaction_edge)
    interaction_edge = [[], []]
    # for i in range(len(train.data)):
        # interaction_edge[i][0].append()
    f.close()
    return train,test,train_csr,train_interaction_edge,test_interaction_edge

def get_interaction():
    long_tail = {}
    api_interatrion_tine = [0 for _ in range(956)]
    sum = 0
    path = '../dataset/'
    with open(path + 'train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')  #  strip表示去除字符串开头和结尾的空白字符（包括其它的特殊符号利于 \n \t）
            items = line[1:]
            for item in items:
                api_interatrion_tine[int(item)] += 1
    for i in range(len(api_interatrion_tine)):
        print(i,api_interatrion_tine[i],'\n')
        if api_interatrion_tine[i] < 2:
            long_tail[i] = 1
            sum +=1
    print(sum)
    return long_tail

if __name__ == '__main__':
    get_bert_emb()
    get_lightgcl_data()
