import time
import datetime

import networkx as nx
from ripple_net.data.brick_data.generate_KG import generate_kg

from utils import tools
import preprocess
import random
import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

from src.pytorchtools import EarlyStopping
import statistics
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from data_loader import load_data
import torch
from src_final.my_model import RippleNet, get_feed_dict, evaluation
import json
import pandas as pd
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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

    # 对数据进行降维处理
    data.fillna(0, inplace=True)
    x = data[[col for col in data.columns if col != 'item_id']].to_numpy()

    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(x)

    # PCA
    print('使用pca降维特征嵌入的数据......')
    rate = args.using_pca

    pca = PCA(n_components=rate)  # 保证降维后的数据保持90%的信息
    pca.fit(x)
    tmp = pca.transform(x)
    print(f'保留{rate}信息降维后的数据维度为:{tmp.shape[1]},为降维前的{tmp.shape[1] / (data.shape[1] - 1)}')
    # if args.item_update_mode == 'plus_transform_with_features':  # 更改模型节点向量维度,维度和pca维度一致
    #     args.dim = tmp.shape[1]
    #     print(args.dim)
    print(f'原先数据的维度为:{data.shape[1] - 1}')
    df = pd.DataFrame(tmp)
    df['item_id'] = data['item_id']
    df['item_id'] = df['item_id'].map(entity2id)
    df.set_index('item_id', drop=True, inplace=True)
    args.add_feat = df.shape[1]
    item_feat_dict = df.T.to_dict('list')

    for key, val in item_feat_dict.items():
        item_feat_dict[key] = torch.FloatTensor(val).to(device=device)

    return item_feat_dict


def train(args, data_info):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    print(f'entity num : {n_entity}')
    print(f'relation num : {n_relation}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    args.item_feat = load_items_feat(args, device)

    writer = SummaryWriter(log_dir=args.log_dir, comment=f'{args.tensor_board_comment}')
    model = RippleNet(args, n_entity, n_relation)
    args.item_feat = load_items_feat(args, device)

    model = model.to(device)
    # 初始化优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    # 初始化早停策略
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    history_eval_acc = []
    history_eval_auc = []
    for epoch in range(args.n_epoch):
        final_loss = 0
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]
            optimizer.zero_grad()  # 不同batch之间清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            final_loss += loss.item()
            start += args.batch_size

        # 可视化结果
        train_auc, train_acc, loss_list = evaluation(args, model, train_data, ripple_set, args.batch_size)
        writer.add_scalars(f'Loss/train',
                           {'loss': loss_list[0], 'base loss': loss_list[1], 'kge loss': loss_list[2],
                            'l2 loss': loss_list[3]}, epoch)  # tensorboard
        train_loss = loss_list[0]

        eval_auc, eval_acc, eval_loss_list = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        history_eval_acc.append(eval_acc)
        history_eval_auc.append(eval_auc)
        writer.add_scalars(f'Loss/eval',
                           {'loss': eval_loss_list[0], 'base loss': eval_loss_list[1], 'kge loss': eval_loss_list[2],
                            'l2 loss': eval_loss_list[3]}, epoch)  # tensorboard
        eval_loss = eval_loss_list[0]

        test_auc, test_acc, test_loss_list = evaluation(args, model, test_data, ripple_set, args.batch_size)
        writer.add_scalars(f'Loss/test',
                           {'loss': test_loss_list[0], 'base loss': test_loss_list[1], 'kge loss': test_loss_list[2],
                            'l2 loss': test_loss_list[3]}, epoch)  # tensorboard
        test_loss = test_loss_list[0]

        writer.add_scalars(f'Loss/all', {'train': train_loss, 'eval': eval_loss, 'test': test_loss},
                           epoch)  # tensorboard
        writer.add_scalars(f'Result/ACC', {'train': train_acc, 'eval': eval_acc, 'test': test_acc}, epoch)
        writer.add_scalars(f'Result/AUC', {'train': train_auc, 'eval': eval_auc, 'test': test_auc}, epoch)

        # print( f'第{epoch}次,eval_acc:{eval_acc}\t,train_acc:{train_acc}\t,eval_auc:{eval_auc}\t,train_auc:{train_auc},'
        #        f'train loss:{train_loss.__round__(4)},eval loos:{eval_loss}')
        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
              % (epoch, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(eval_loss_list[0], model)
        # early_stopping(eval_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 可视化模型
    writer.close()
    print('训练完成')
    print('最好的验证集准确率为:', round(max(history_eval_acc), 3))
    print('最好的验证集auc为:', round(max(history_eval_auc), 3))

    return round(max(history_eval_acc), 3), round(max(history_eval_auc), 3)


def init():
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brick_data', help='which dataset to use')
    parser.add_argument('--date', type=str, default='1231', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--patience', type=int, default=10, help='早停次数')
    parser.add_argument('--tensor_board_comment', type=str, default='序列实验', help='可视化工具文件的后缀')
    parser.add_argument('--log_dir', type=str, default='./runs/序列实验', help='可视化工具文件文件名')
    parser.add_argument('--add_feat', type=int, default=0, help='商品特征维度')
    parser.add_argument('--important_sample', type=bool, default=False, help='是否重点采样')
    parser.add_argument('--attr_filter', type=list, default=[], help='是否筛选边的类型')

    """ 
    经过调参后,基本确定是最好的参数,不要轻易修改
    """
    # parser.add_argument('--debug', type=bool, default=False, help='是否开启调试模型')
    parser.add_argument('--debug', type=bool, default=True, help='是否开启调试模型')
    parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.1, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform_with_features', help='')
    parser.add_argument('--using_pca', type=int, default=64, help='特征嵌入钱做pca降维,数值代表降维后的数据维度')
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

    # preprocess.preprocess() # 实验中生成一次即可
    # 设置随机数种子
    setup_seed(2021)
    data_info = load_data(args)

    # 引入一个time模块， * 表示time模块的所有功能，
    # 作用： 可以统计程序运行的时间

    begin_time = time.time()

    acc, auc = train(args, data_info)

    end_time = time.time()
    run_time = end_time - begin_time
    print('训练时间：', run_time)  # 该循环程序运行时间： 1.4201874732

    return acc, auc, run_time


def get_kg(upper_time):
    # 调用结果
    # upper_time = datetime.datetime.strptime(time, "%Y/%m/%d")

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    # parser.add_argument('--date', type=str, default='0514', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--date', type=str, default='1231', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--upper_time', default=upper_time, help='保存图谱知识的最大日期')
    parser.add_argument('--save2neo', default=False, help='是否保存图谱到neo4j')
    parser.add_argument('--recommender_path', type=str, default='../../recommender/gobricks_social/', help='推荐文件路径')
    parser.add_argument('--save_path', type=str, default='../data/brick_data/', help='文件保存路径')
    parser.add_argument('--reverse', type=bool, default=True, help='是否添加反转关系')
    # parser.add_argument('--model', type=str, default='default', help='使用什么模型')
    # parser.add_argument('--save_path', type=str, default='../../../GATNE/map/gobricks_builds/', help='gatne模型数据保存路径')
    # parser.add_argument('--data_path', type=str, default='../../../recommender/gobricks_social/features/', help='文件读取路径')
    args = parser.parse_args(args=[])

    kg = generate_kg(args)
    return kg


def analyz_kg(kg):
    # 通过networkx获取图谱参数
    file = '../data/brick_data/kg_final_no_relation.txt'
    edit_kg_file = pd.read_csv(kg, sep='\t', names=['head', 'relation', 'tail'])
    edit_kg_file[['head', 'tail']].to_csv(file, index=False, header=False, sep='\t')
    G = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int, )  # 创建图
    data = pd.read_csv(kg, header=None, sep='\t')
    # 网络密度、网络直径、节点数量、知识数量
    # print(nx.density(G))
    # print(nx.diameter(G))
    # print(len(G.nodes()))
    # print(G.size())
    result = {
        'density': nx.density(G),
        'diameter': nx.diameter(G),
        'node_num': len(G.nodes()),
        'edge_num': G.size(),  # 没有考虑边的类型
        'edge_class_num': len(data[1].unique()),
        'knowledge_num': data.shape[0],
    }
    print(result)
    return result


def pipeline():
    print(os.getcwd())
    if os.path.exists('kg_sequence_result.csv'):
        result = pd.read_csv('kg_sequence_result.csv')
    else:
        result = ['time', 'density', 'diameter', 'node_num', 'edge_num', 'knowledge_num', 'acc', 'auc',
                  'train_time']
    # test
    if result.shape[0] == 0:
        kg_time = "2021/12/01"
    else:
        kg_time = result.loc[result.shape[0]]['time'].replace('-', '/')

    gap = datetime.datetime.strptime("2021/01/31", "%Y/%m/%d") - datetime.datetime.strptime(kg_time, "%Y/%m/%d")
    for step in range(0, gap.days, 7):
        upper_time = datetime.datetime.strptime(kg_time, "%Y/%m/%d") + datetime.timedelta(days=step)
        print(upper_time)
        kg = get_kg(upper_time)
        preprocess.preprocess()  # 数据预处理
        res = analyz_kg('../data/brick_data/kg_final.txt')
        acc, auc, train_time = init()
        # 保存结果
        result.loc[result.shape[0]] = {
            'time': upper_time,
            'density': res['density'],
            'diameter': res['diameter'],
            'node_num': res['node_num'],
            'edge_num': res['edge_num'],
            'knowledge_num': res['knowledge_num'],
            'acc': acc,
            'auc': auc,
            'train_time': train_time
        }
        result.to_csv('kg_sequence_result.csv', encoding='utf_8_sig', index=False)


def one_time():
    '''
    一次行使用的启动函数，因为脚本运动多个模型会出错。
    '''
    print(os.getcwd())
    if os.path.exists('kg_sequence_result.csv'):
        result = pd.read_csv('kg_sequence_result.csv')
    else:
        result = pd.DataFrame(
            columns=['time', 'density', 'diameter', 'node_num', 'edge_num', 'knowledge_num', 'acc', 'auc',
                     'train_time'])

    if result.shape[0] == 0:
        time = "2021/01/01"
        upper_time = datetime.datetime.strptime(time, "%Y/%m/%d")
    else:
        time = result.at[result.shape[0] - 1, 'time'].split(' ')[0].replace('-', '/')
        upper_time = datetime.datetime.strptime(time, "%Y/%m/%d") + datetime.timedelta(days=7)

    print(upper_time)
    kg = get_kg(upper_time)
    preprocess.preprocess()  # 数据预处理
    res = analyz_kg('../data/brick_data/kg_final.txt')
    acc, auc, train_time = init()
    # 保存结果
    result.loc[result.shape[0]] = {
        'time': upper_time,
        'density': res['density'],
        'diameter': res['diameter'],
        'node_num': res['node_num'],
        'edge_num': res['edge_num'],
        'knowledge_num': res['knowledge_num'],
        'acc': acc,
        'auc': auc,
        'train_time': train_time
    }
    result.to_csv('kg_sequence_result.csv', encoding='utf_8_sig', index=False)


def get_knowledge_num():
    """
    获取图谱知识数量
    """
    result = pd.read_csv('kg_sequence_result.csv')
    result['knowledge_num'] = None

    for i in range(result.shape[0]):
        time = result.at[i, 'time'].split(' ')[0].replace('-', '/')
        upper_time = datetime.datetime.strptime(time, "%Y/%m/%d")
        print(upper_time)
        kg = get_kg(upper_time)
        preprocess.preprocess()  # 数据预处理
        res = analyz_kg('../data/brick_data/kg_final.txt')
        result.at[i, 'knowledge_num'] = res['knowledge_num']
        result.to_csv('kg_sequence_result.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    # pipeline()
    one_time()
    # get_knowledge_num()
    os.system('say "your program has finished"')
