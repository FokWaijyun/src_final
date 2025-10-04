import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import json


class RippleNet(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation)

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.criterion = nn.BCELoss()

        # 判断是否要加入商品特征进行训练
        # todo 添加激活函数
        if args.add_feat == 0:
            self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        else:
            self.transform_matrix = nn.Linear(self.dim + args.add_feat, self.dim, bias=False)
            self.transform_matrix = nn.Sequential(
                nn.Linear(self.dim + args.add_feat,(self.dim + args.add_feat)//2, bias=False),
                nn.ReLU(),
                torch.nn.Linear((self.dim + args.add_feat)//2,  self.dim),
            )
        print('打印模型参数')
        print('{0:20} ==> {1:20}'.format('dim', self.dim))
        print('{0:20} ==> {1:20}'.format('add_feat', args.add_feat))

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.item_feat = args.item_feat  # 特征字典文件

    def forward(
            self,
            items: torch.LongTensor,
            labels: torch.LongTensor,
            memories_h: list,
            memories_r: list,
            memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []

        # 读取三元组知识向量
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            # try:
            #     t_emb_list.append(self.entity_emb(memories_t[i]))
            # except BaseException as e:
            #     print(e)

        # 计算用户兴趣传播
        o_list, item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings, items
        )
        # 预测 o_list:绿色小方块
        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )

        return_dict["scores"] = scores
        return_dict["o_list"] = o_list
        return_dict["entity_emb"] = self.entity_emb

        return return_dict

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings, items):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]  输入的点击的商品
            v = torch.unsqueeze(item_embeddings, dim=2)

            # [batch_size, n_memory]  对比相似都
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            if len(probs.shape) == 1:
                probs = torch.reshape(input=probs, shape=(1, -1))

            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o, items)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o, items):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        elif self.item_update_mode == "plus_transform_with_features":
            # 拼接向量(item特征向量)
            item_input = []
            for i, item_index in enumerate(items):
                tmp = torch.cat((item_embeddings[i], self.item_feat[int(item_index)]), 0)
                item_input.append(tmp)
            item_input = torch.stack(item_input)
            item_embeddings = self.transform_matrix(item_input)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())  # 二分类交叉熵

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def predict(self, item_embeddings, o_list):
        '''
        o_list 是示意图中绿色矩阵的部分,使用了ATTENTION之后的兴趣涟漪之后的向量
        '''
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]  # todo 这里可以优化

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)  # 这个就是用户点击商品的概率
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except:
            auc = 0
        predictions = [1 if i >= 0.5 else 0 for i in scores]  # 卡阈值
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc, return_dict


def get_feed_dict(args, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r, memories_t


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()  # 切换到评价模型

    loss = 0
    base_loss = 0
    kge_loss = 0
    l2_loss = 0
    length = data.shape[0] // args.batch_size
    while start < data.shape[0]:
        auc, acc, return_dict = model.evaluate(*get_feed_dict(args, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size

        loss += return_dict["loss"].item()
        base_loss += return_dict["base_loss"].item()
        kge_loss += return_dict["kge_loss"].item()
        l2_loss += return_dict["l2_loss"].item()
    model.train()  # 切换到训练模型

    return float(np.mean(auc_list)), float(np.mean(acc_list)), [loss / length, base_loss / length, kge_loss / length,
                                                                l2_loss / length]
