'''
控制组实验，最原始的代码
'''

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

    args.item_feat = None

    writer = SummaryWriter(log_dir=args.log_dir, comment=f'{args.tensor_board_comment}')
    model = RippleNet(args, n_entity, n_relation)
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

    return round(max(history_eval_acc), 3)


if __name__ == '__main__':
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brick_data', help='which dataset to use')
    parser.add_argument('--date', type=str, default='1231', help='用户切分时用到的数据文件的后缀日期,在用户切分时使用')
    parser.add_argument('--patience', type=int, default=10, help='早停次数')
    parser.add_argument('--tensor_board_comment', type=str, default='重要性采样', help='可视化工具文件的后缀')
    parser.add_argument('--log_dir', type=str, default='./runs/重要性采样', help='可视化工具文件文件名')
    parser.add_argument('--add_feat', type=int, default=0, help='商品特征维度')
    parser.add_argument('--important_sample', type=bool, default=True, help='是否重点采样')
    parser.add_argument('--attr_filter', type=list, default=[], help='是否筛选边的类型')

    """ 
    经过调参后,基本确定是最好的参数,不要轻易修改
    """
    # parser.add_argument('--debug', type=bool, default=False, help='是否开启调试模型')
    parser.add_argument('--debug', type=bool, default=True, help='是否开启调试模型')
    parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.1, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform', help='')
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

    if not args.debug:  # 删除调试时的中间文件
        print('请重新生成图谱知识！')
        try:
            os.remove(f"kg_{args.date}.npy")
            os.remove(f"ripple_set_{args.date}.npy")
            os.remove(f"test_{args.date}.npy")
        except:
            print('要删除的文件不存在')
        preprocess.preprocess()

    # 设置随机数种子
    # setup_seed(2021)
    data_info = load_data(args)

    # 引入一个time模块， * 表示time模块的所有功能，
    # 作用： 可以统计程序运行的时间
    from time import *

    begin_time = time()

    train(args, data_info)

    end_time = time()
    run_time = end_time - begin_time
    print('训练时间：', run_time)  # 该循环程序运行时间： 1.4201874732

