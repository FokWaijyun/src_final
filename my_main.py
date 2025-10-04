from utils import tools
import preprocess
from ripple_net.data.brick_data import generate_KG
import random
import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from merge_xgboost import train_model

from src.pytorchtools import EarlyStopping
import statistics
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from data_loader import load_data
import torch
from src_reconstitution.my_model import RippleNet, get_feed_dict, evaluation
import json
import pandas as pd
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_user_interesting():
    """
    # 根据用户的兴趣涟漪向量得到用户对不同作品的兴趣强度
    - 使用SIGMOD函数来标准化这个兴趣强度
    """
    print('开始提取用户的兴趣特征')
    data = pd.read_csv(f'../data/brick_data/item_index2entity_id_rehashed.txt', header=None, sep='\t',
                       encoding='utf_8_sig')
    print('作品的数量:', data.shape[0])
    item_num = data.shape[0]
    item2id = {}
    id2item = {}
    for index, row in tqdm(data.iterrows()):
        item2id[row[0]] = row[1]
        id2item[row[1]] = row[0]
    # %%
    entity_emb = json.load(open(f'entity_emb.json', 'r', encoding='utf8'))
    user2id = json.load(open(f'../data/brick_data/map/user2id.json', 'r', encoding='utf8'))
    id2user = {id: user for user, id in user2id.items()}
    user_interact_emb = json.load(open(f'user_interact_emb.json', 'r', encoding='utf8'))

    """ 
    # 用户对作品的兴趣强度
    - 用户涟漪求和
    - 或许有其他方法: 平均
    - 获取item的emb
    - 计算用户对作品的兴趣强度
    - 论文中是将两个向量点乘后, 逐个相加, 使用sigmoid函数计算强度的
    - 后续可以考虑使用其他方法来做强度计算
    - {user: {item1: score, item2: score, ......}}
    """
    train = pd.read_csv(f'train_{args.date}.csv', encoding='utf_8_sig')
    test = pd.read_csv(f'test_{args.date}.csv', encoding='utf_8_sig')
    data = pd.concat([train, test])
    user_hop_emb = {}
    for id, embs in tqdm(user_interact_emb.items()):
        user_hop_emb[id2user[int(id)]] = torch.Tensor(embs[0]).add(torch.Tensor(embs[1]))

    user2item_score = {}
    for user, hop_emb in tqdm(user_hop_emb.items()):
        user2item_score[user] = {}
        for item_id, item_name in id2item.items():
            tmp = (hop_emb * torch.Tensor(entity_emb[item_name])).sum()
            score = torch.sigmoid(tmp)
            user2item_score[user][item_name] = score.data.item()
    with open('user2item_score.json', 'w', encoding='utf8') as fp:
        json.dump(user2item_score, fp, ensure_ascii=False)


def load_entity_pretrain(args, entity_num):
    """
    读取节点预训练向量
    - 载入预训练节点向量
        - 计算节点特征向量: 将不同边属性下的节点向量求平均
    """
    node2id = json.load(open(f'../data/brick_data/map/entity_id2index.json', 'r', encoding='utf8'))
    node_emb = np.load(f'../data/brick_data/emb/final_model.npy', allow_pickle=True)

    # 求节点平均向量
    emb_temp = None
    for key in node_emb[()].keys():
        if not emb_temp:
            emb_temp = node_emb[()][key].copy()
            continue
        for k, val in emb_temp.items():
            emb_temp[k] += node_emb[()][key][k]

    # 平均向量
    for k, val in emb_temp.items():
        emb_temp[k] /= len(node_emb[()].keys())

    emb = emb_temp
    print('对齐模型特征维度')
    args.dim = emb['作品类别.unknown'].shape[0]
    model_node_emb = np.ones([entity_num, args.dim])

    # 将节点向量对齐到模型的节点上
    for key, val in tqdm(emb.items()):
        index2model_entity = node2id[key]
        model_node_emb[index2model_entity] = val

    return model_node_emb


def generate_entity_emb(args, entity_num):
    """
    利用节点自身的特征构建节点向量
    不同类型的节点,使用merge的方式合并起来
    - 暂时图谱只有三种类型的节点,用户,商品和商品类别
    """
    print('构建模型节点特征......')
    features_path = '../../recommender/gobricks_social/features/'
    node2id = json.load(open(f'../data/brick_data/map/entity_id2index.json', 'r', encoding='utf8'))
    id2node = {_id: node for node, _id in node2id.items()}
    items_features = pd.read_csv(f'{features_path}items_features_{args.date}.csv', encoding='utf_8_sig')
    items = items_features['item_id'].tolist()
    items_features.set_index('item_id', drop=True, inplace=True)
    users_features = pd.read_csv(f'{features_path}users_features_{args.date}.csv', encoding='utf_8_sig')
    users = users_features['user_id'].tolist()
    users_features.set_index('user_id', drop=True, inplace=True)

    # 用户商品特征各,类别特征加上一列
    dim = (users_features.shape[1]) + (items_features.shape[1]) + 1
    args.dim = dim
    node_emb = np.ones([entity_num, dim])

    for i in tqdm(range(entity_num)):
        node = id2node[i]
        if node in users:
            u_feat = users_features.loc[node].tolist()
            i_feat = [0] * items_features.shape[1]
            c_feat = [0]
        elif node in items:
            u_feat = [0] * users_features.shape[1]
            i_feat = items_features.loc[node].tolist()
            c_feat = [0]
        else:
            u_feat = [0] * users_features.shape[1]
            i_feat = [0] * items_features.shape[1]
            c_feat = [1]
        node_emb[i] = u_feat + i_feat + c_feat

    return node_emb


def load_items_feat(args, device):
    print('开始载入商品特征')
    features_path = '../../recommender/gobricks_social/features/'
    entity2id = json.load(open(f'../data/brick_data/map/entity_id2index.json', 'r', encoding='utf8'))
    data = pd.read_csv(f'{features_path}items_features_{args.date}.csv', encoding='utf_8_sig')

    print('删除空值超过一半的特征')
    tmp = data.isna().sum()
    del_features = []
    for key, value in tmp.items():
        if value > data.shape[0] // 2:
            del_features.append(key)
    for col in del_features:
        del data[col]

    # 没有进行降维
    data.fillna(0, inplace=True)
    data['item_id'] = data['item_id'].map(entity2id)
    data.set_index('item_id', drop=True, inplace=True)
    args.add_feat = data.shape[1]
    item_feat_dict = data.T.to_dict('list')

    # 对数据进行降维处理
    # map.fillna(0, inplace=True)
    # x = map[[col for col in map.columns if col != 'item_id']].to_numpy()
    #
    # # feature normalization (feature scaling)
    # X_scaler = StandardScaler()
    # x = X_scaler.fit_transform(x)
    #
    # # PCA
    # rate = 0.8
    # pca = PCA(n_components=rate)  # 保证降维后的数据保持90%的信息
    # pca.fit(x)
    # tmp = pca.transform(x)
    # print(f'保留{rate}信息降维后的数据维度为:{tmp.shape[1]},为降维前的{tmp.shape[1] / (map.shape[1] - 1)}')
    # print(f'原先数据的维度为:{map.shape[1] - 1}')
    # df = pd.DataFrame(tmp)
    # df['item_id'] = map['item_id']
    # df['item_id'] = df['item_id'].map(entity2id)
    # df.set_index('item_id', drop=True, inplace=True)
    # args.add_feat = df.shape[1]
    # item_feat_dict = df.T.to_dict('list')

    for key, val in item_feat_dict.items():
        item_feat_dict[key] = torch.FloatTensor(val).to(device=device)

    return item_feat_dict


def load_users_feat(args, device):
    print('开始载入用户特征')
    features_path = '../../recommender/gobricks_social/features/'
    entity2id = json.load(open(f'../data/brick_data/map/entity_id2index.json', 'r', encoding='utf8'))
    data = pd.read_csv(f'{features_path}users_features_{args.date}.csv', encoding='utf_8_sig')

    data.fillna(0, inplace=True)
    df = data
    df['user_id'] = data['user_id']
    # df['user_id'] = df['user_id'].map(entity2id)
    # df.dropna(subset=['user_id'],inplace=True)
    df.set_index('user_id', drop=True, inplace=True)

    args.add_feat += df.shape[1]
    user_feat_dict = df.T.to_dict('list')

    for key, val in user_feat_dict.items():
        user_feat_dict[key] = torch.FloatTensor(val).to(device=device)

    return user_feat_dict


def train(args, data_info):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    print(f'entity num : {n_entity}')
    print(f'relation num : {n_relation}')
    all_data = np.concatenate((train_data, eval_data), axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    args.item_feat = 0
    if args.entity_pretrain:
        item_feat = load_items_feat(args, device)
        # user_feat = load_users_feat(args, device)
        args.item_feat = item_feat
        # args.user_feat = user_feat

    writer = SummaryWriter(log_dir=args.log_dir, comment=f'{args.tensor_board_comment}')
    if args.entity_pretrain == 'GATNE':
        print('使用GATNE方式加载预训练节点向量......')
        model_node_emb = load_entity_pretrain(args, n_entity)
        model = RippleNet(args, n_entity, n_relation)
        model.entity_emb.weight.data.copy_(torch.from_numpy(model_node_emb))
    elif args.entity_pretrain == 'self_features':
        print('利用节点特征构建节点向量......')
        node_emb = generate_entity_emb(args, n_entity)
        model = RippleNet(args, n_entity, n_relation)
        model.entity_emb.weight.data.copy_(torch.from_numpy(node_emb))
    else:
        model = RippleNet(args, n_entity, n_relation)

    model = model.to(device)
    # 初始化优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    # 初始化早停策略
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    add_flag = 0
    user_interact_emb = {}  #
    history_eval_acc = []
    for epoch in range(args.n_epoch):
        final_loss = 0
        length = train_data.shape[0] // args.batch_size
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]
            # 保存用户兴趣涟漪向量
            for index, val in enumerate(train_data[start:start + args.batch_size]):
                user = int(val[0])
                user_interact_emb[user] = [return_dict['o_list'][i].detach().cpu().numpy()[index].tolist() for i in
                                           range(args.n_hop)]

            optimizer.zero_grad()  # 不同batch之间清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            final_loss += loss.item()
            start += args.batch_size

        # print(f'第{epoch}次平均训练误差为{final_loss / length}')
        # writer.add_scalar('Loss/train', final_loss / length, epoch)  # tensorboard

        # 可视化结果
        train_auc, train_acc, loss_list = evaluation(args, model, train_data, ripple_set, args.batch_size)
        writer.add_scalars(f'{args.dataset}_Loss/train',
                           {'loss': loss_list[0], 'base loss': loss_list[1], 'kge loss': loss_list[2],
                            'l2 loss': loss_list[3]}, epoch)  # tensorboard
        train_loss = loss_list[0]
        eval_auc, eval_acc, eval_loss_list = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        history_eval_acc.append(eval_acc)
        writer.add_scalars(f'{args.dataset}_Loss/eval',
                           {'loss': eval_loss_list[0], 'base loss': eval_loss_list[1], 'kge loss': eval_loss_list[2],
                            'l2 loss': eval_loss_list[3]}, epoch)  # tensorboard
        eval_loss = eval_loss_list[0]
        writer.add_scalars(f'{args.dataset}_Loss/train&eval', {'train_loss': train_loss, 'eval_loss': eval_loss},
                           epoch)  # tensorboard
        writer.add_scalars(f'{args.dataset}_AUC&ACC/ACC/train&eval',
                           {'train_acc': train_acc, 'eval_acc': eval_acc},
                           epoch)  # tensorboard
        writer.add_scalars(f'{args.dataset}_AUC&ACC/AUC/train&eval',
                           {'train_auc': train_auc, 'eval_auc': eval_auc},
                           epoch)  # tensorboard

        print(
            f'第{epoch}次,eval_acc:{eval_acc}\t,train_acc:{train_acc}\t,eval_auc:{eval_auc}\t,train_auc:{train_auc},train loss:{train_loss.__round__(4)},eval loos:{eval_loss}')
        # 查看模型是否有更新除了商品之外的向量
        # print(model.entity_emb.weight[0])
        # print(model.entity_emb.weight[-1])

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(eval_loss_list[0], model)
        # early_stopping(eval_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    evaluation(args, model, all_data, ripple_set, args.batch_size, save_flag=True)
    # 保存节点向量
    node2id = json.load(open(f'../data/{args.dataset}/map/entity_id2index.json', 'r', encoding='utf8'))
    id2node = {_id: key for key, _id in node2id.items()}
    node_emb = {}
    for index, emb in enumerate(return_dict['entity_emb'].weight.tolist()):
        node_emb[id2node[index]] = emb

    with open('entity_emb.json', 'w', encoding='utf8') as fp:
        json.dump(node_emb, fp, ensure_ascii=False)
    # 保存用户兴趣涟漪的相关数据
    with open('user_interact_emb.json', 'w', encoding='utf8') as fp:
        json.dump(user_interact_emb, fp, ensure_ascii=False)
    # 可视化模型
    writer.close()
    print('训练完成')
    print('最好的验证集准确率为:', round(max(history_eval_acc), 3))
    # auc绘制
    return round(max(history_eval_acc), 3)


def pipeline_train():
    """
    pileline 训练模块
    后面接了精排模型：xgboost
    :return:
    """
    np.random.seed(555)
    # """:arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brick_data', help='which dataset to use')
    parser.add_argument('--date', type=str, default='0823', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--user_split', type=bool, default=False, help='which way to split map')
    # parser.add_argument('--entity_pretrain', type=str, default='GATNE', help='节点向量预训练方式')
    parser.add_argument('--entity_pretrain', type=str, default='none', help='节点向量预训练方式')
    parser.add_argument('--patience', type=int, default=5, help='早停次数')
    parser.add_argument('--tensor_board_comment', type=str, default='', help='可视化工具文件的后缀')
    # parser.add_argument('--log_dir', type=str, default='runs/exp5_添加节点特征', help='可视化工具文件文件名')
    parser.add_argument('--log_dir', type=str, default='', help='可视化工具文件文件名')
    parser.add_argument('--add_feat_flag', type=bool, default=True, help='训练商品向量时是否加入商品特征进行训练')
    parser.add_argument('--add_feat', type=int, default=0, help='商品特征维度')
    parser.add_argument('--important_sample', type=bool, default=False, help='商品特征维度')
    # parser.add_argument('--add_user_feat', type=int, default=0, help='训练商品向量时是否加入商品特征进行训练')

    build_info = ['brick.build.belong.classes', 'brick.build.classes.include']
    user_behavior_info = ['user.buy.build', 'user.add_cart.build', 'user.collect.build', 'user.thumb_up.build',
                          'user.comment.build', 'user.click.build', 'build.buy.user', 'build.add_cart.user',
                          'build.collect.user', 'build.thumb_up.user', 'build.comment.user', 'build.click.user',
                          'user.design.build', 'user.interact.build', 'build.interact.user']
    component_info = ['component.belong.classes', 'classes.include.component', 'build.include.component',
                      'component.include.build']
    social_info = ['user.focus.user']

    parser.add_argument('--attr_filter', type=list, default=[], help='是否筛选边的类型')
    # parser.add_argument('--attr_filter', type=list, default=build_info+user_behavior_info, help='是否筛选边的类型')
    # parser.add_argument('--attr_filter', type=list, default=user_behavior_info, help='是否筛选边的类型')
    # parser.add_argument('--attr_filter', type=list, default=component_info+user_behavior_info+build_info, help='是否筛选边的类型')

    """ 
    经过调参后,基本确定是最好的参数,不要轻易修改
    """
    parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
    # parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.1, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    # parser.add_argument('--item_update_mode', type=str, default='plus_transform_with_features', help='')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')

    # parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    print('cuda:', torch.cuda.is_available())
    parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(), help='whether to use gpu')
    args = parser.parse_args(args=[])
    print('ripple net 实验参数:')
    for key, val in args.__dict__.items():
        if key not in ['cuda', 'dataset', 'date', 'user_split', 'entity_pretrain', 'tensor_board_comment', 'log_dir',
                       'user_cuda', 'attr_filter']:
            print('{0:20} ==> {1:20}'.format(key, val))
    print()

    # 设置随机数种子
    # setup_seed(2021)
    relation_filter_list = {
        '不筛选': [],
        # '作品类': build_info,
        # '行为类': user_behavior_info,
        # '作品类+行为类': build_info + user_behavior_info,
        # '作品类+零件类': build_info + component_info,
        # '行为类+零件类': user_behavior_info + component_info,
    }
    # res = {}
    res = json.load(open(f'exp_precision.json', 'r', encoding='utf8'))
    xgb_res = json.load(open(f'exp_xgb.json', 'r', encoding='utf8'))
    # 重复实验
    for key, val in relation_filter_list.items():
        xgb_precision = train_model(args)
        xgb_res[key] = xgb_precision
        print('test xgboost')
        break
        # try:
        start_time = time.time()  # 转化格
        args.attr_filter = val
        best_eval_acc_list = []
        exp_time = 10
        for _ in range(exp_time):
            print(f'{key},第{_}次实验')
            # train(args, data_info)
            data_info = load_data(args)
            best_eval_acc_list.append(train(args, data_info))

        for acc in best_eval_acc_list:
            print(acc)

        print(f'{exp_time}次实验平均准确率:', statistics.mean(best_eval_acc_list))
        # 提取用户的对套件的兴趣作为推荐模型的输入特征
        extract_user_interesting()
        # 进行xgboost预测嵌入用户特征
        xgb_precision = train_xgb(args)
        xgb_res[key] = xgb_precision

        # 这是计算此时运行耗费多长时间，特意转化为 时:分:秒
        end_time = time.time()  # 转化格式
        Total_time = end_time - start_time
        m, s = divmod(Total_time, 60)
        h, m = divmod(m, 60)
        print("共耗时===>%d时:%02d分:%02d秒" % (h, m, s))

        # 保存n次实验结果
        res[key] = best_eval_acc_list
        with open('exp_precision.json', 'w', encoding='utf8') as fp:
            json.dump(res, fp, ensure_ascii=False)
    # except BaseException as e:
    #     print(e)

    with open('exp_precision.json', 'w', encoding='utf8') as fp:
        json.dump(res, fp, ensure_ascii=False)

    print(res)


def kg_split_train():
    """
    图谱时间训练模块
    :return:
    """
    np.random.seed(555)
    # """:arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brick_data', help='which dataset to use')
    parser.add_argument('--date', type=str, default='0823', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--user_split', type=bool, default=False, help='which way to split map')
    parser.add_argument('--entity_pretrain', type=str, default='none', help='节点向量预训练方式')
    parser.add_argument('--patience', type=int, default=5, help='早停次数')
    parser.add_argument('--tensor_board_comment', type=str, default='', help='可视化工具文件的后缀')
    # parser.add_argument('--log_dir', type=str, default='runs/exp5_添加节点特征', help='可视化工具文件文件名')
    parser.add_argument('--log_dir', type=str, default='', help='可视化工具文件文件名')
    parser.add_argument('--add_feat_flag', type=bool, default=True, help='训练商品向量时是否加入商品特征进行训练')
    parser.add_argument('--add_feat', type=int, default=0, help='商品特征维度')
    parser.add_argument('--important_sample', type=bool, default=False, help='商品特征维度')
    # parser.add_argument('--add_user_feat', type=int, default=0, help='训练商品向量时是否加入商品特征进行训练')
    parser.add_argument('--attr_filter', type=list, default=[], help='是否筛选边的类型')

    """ 
    经过调参后,基本确定是最好的参数,不要轻易修改
    """
    parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
    # parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.1, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=128, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform_with_features', help='')
    # parser.add_argument('--item_update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')

    ''' 其他文件的设置 '''
    parser.add_argument('--recommender_path', type=str, default='../../recommender/gobricks_social/', help='推荐文件路径')
    parser.add_argument('--save_path', type=str, default='', help='数据生成文件保存路径')
    parser.add_argument('--debug', type=bool, default='True', help='数据生成文件保存路径')

    # parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    print('cuda:', torch.cuda.is_available())
    parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(), help='whether to use gpu')
    args = parser.parse_args(args=[])
    print('ripple net 实验参数:')
    for key, val in args.__dict__.items():
        if key not in ['cuda', 'dataset', 'date', 'user_split', 'entity_pretrain', 'tensor_board_comment', 'log_dir',
                       'user_cuda', 'attr_filter']:
            print('{0:20} ==> {1:20}'.format(key, val))

    res = json.load(open(f'kg_split_precision.json', 'r', encoding='utf8'))
    xgb_res = json.load(open(f'kg_split_xgb.json', 'r', encoding='utf8'))
    # todo 不同图谱的实验
    path = '../data/brick_data/'
    times = tools.get_split_time(f'{path}kg_rehashed', gap=30, n_split=3)  # 要先得到全部知识的时间
    print(times)
    for upper_time in times:
        # 生成相关文件
        args.upper_time = upper_time
        # generate_KG.generate_kg(args)
        # todo pagerank文件
        # 数据预处理
        # preprocess.preprocess()
        ### todo 模型训练
        # todo ripple net 模型训练
        data_info = load_data(args)
        print(train(args, data_info))
        print("时间图谱的日期为：",upper_time)
        # todo 下游xgboost模型

        # todo 保存结果
        pass

    print('done')


    # 重复实验
    for key, val in relation_filter_list.items():
        # try:
        start_time = time.time()  # 转化格

        args.attr_filter = val
        best_eval_acc_list = []
        exp_time = 10
        for _ in range(exp_time):
            print(f'{key},第{_}次实验')
            # train(args, data_info)
            data_info = load_data(args)
            best_eval_acc_list.append(train(args, data_info))

        for acc in best_eval_acc_list:
            print(acc)

        print(f'{exp_time}次实验平均准确率:', statistics.mean(best_eval_acc_list))
        # 提取用户的对套件的兴趣作为推荐模型的输入特征
        extract_user_interesting()
        # 进行xgboost预测嵌入用户特征
        xgb_precision = train_xgb(args)
        xgb_res[key] = xgb_precision

        # 这是计算此时运行耗费多长时间，特意转化为 时:分:秒
        end_time = time.time()  # 转化格式
        Total_time = end_time - start_time
        m, s = divmod(Total_time, 60)
        h, m = divmod(m, 60)
        print("共耗时===>%d时:%02d分:%02d秒" % (h, m, s))

        # 保存n次实验结果
        res[key] = best_eval_acc_list
        with open('exp_precision.json', 'w', encoding='utf8') as fp:
            json.dump(res, fp, ensure_ascii=False)
    # except BaseException as e:
    #     print(e)

    with open('exp_precision.json', 'w', encoding='utf8') as fp:
        json.dump(res, fp, ensure_ascii=False)

    print(res)


if __name__ == '__main__':
    kg_split_train()
