import collections
from tqdm import tqdm
from collections import Counter
import json
import os
import numpy as np
import pandas as pd


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        # np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    print(f'用户数量:{n_user}')
    print(f'商品数量:{n_item}')
    return dataset_split(args, rating_np)


def user_list(args):
    """:arg
    用户切分函数,用户topN指标
    """
    path = f'../../recommender/gobricks_social/data/'
    train_user = pd.read_csv(f'{path}train_{args.date}.csv', encoding='utf_8_sig')
    test_user = pd.read_csv(f'{path}test_{args.date}.csv', encoding='utf_8_sig')

    return train_user['relate_user_id'].unique(), test_user['relate_user_id'].unique()


def dataset_split(args, rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    # eval_ratio = 0.2
    # test_ratio = 0.2
    eval_ratio = 0.2
    test_ratio = 0.0
    n_ratings = rating_np.shape[0]
    # 用户切分数据
    if args.user_split:
        print('user splitting ......')
        train_user, test_user = user_list(args)
        user2id = json.load(open(f'../data/{args.dataset}/data/user2id.json', encoding='utf8'))
        train_map_obj = map(lambda user: user2id[user], train_user)
        eval_map_obj = map(lambda user: user2id[user], test_user)
        rating_pd = pd.DataFrame(rating_np, columns=['user_id', 'item_id', 'rating'])

        eval_users = list(eval_map_obj)
        eval_indices = rating_pd[rating_pd['user_id'].isin(eval_users)].index
        train_users = list(train_map_obj)
        train_indices = rating_pd[rating_pd['user_id'].isin(train_users)].index
        test_indices = []
        user_indices = train_indices.tolist()
        user_indices.extend(eval_indices.tolist())
    else:
        eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))
        user_indices = range(n_ratings)

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in user_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # 保存数据集给排序模型做训练
    data = pd.read_csv(f'ratings.csv', encoding='utf_8_sig', header=None)
    data.loc[train_indices].to_csv(f'train_{args.date}.csv', index=False, header=False, encoding='utf_8_sig')
    data.loc[eval_indices].to_csv(f'test_{args.date}.csv', index=False, header=False, encoding='utf_8_sig')
    # 打印相关信息
    print(f'数据集数量: train:{len(train_indices)},test:{len(eval_indices)}, test:{len(test_indices)}')
    print(
        f'正负样本比例: train:{Counter(data.loc[train_indices][2])},test:{Counter(data.loc[eval_indices][2])}, test:{Counter(data.loc[test_indices][2])}')

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')
    np.random.seed(2021)

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        # np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_ripple_set(args, kg, user_history_dict):
    '''
    固定类型的边
    '''
    node_important = json.load(open(f'../data/brick_data/data/node_important.json', 'r', encoding='utf8'))
    node_important = {int(key): val for key, val in node_important.items()}
    # 读取关系映射
    relation2id = json.load(open(f'../data/brick_data/data/relation_id2index.json', 'r', encoding='utf8'))
    relation_filter = []
    for relation in args.attr_filter:
        if relation in relation2id:
            relation_filter.append(relation2id[relation])

    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        if user == 0:
            print('in')
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                temp_head = []
                temp_relation = []
                temp_tail = []
                for tail_and_relation in kg[entity]:
                    # todo 重点采样
                    if not relation_filter or tail_and_relation[1] in relation_filter:
                        if args.important_sample:
                            temp_relation.append(tail_and_relation[1])
                        else:
                            memories_r.append(tail_and_relation[1])
                    else:
                        continue
                    if args.important_sample:
                        temp_head.append(entity)
                        temp_tail.append(tail_and_relation[0])
                    else:
                        memories_h.append(entity)
                        memories_t.append(tail_and_relation[0])

                if args.important_sample:
                    # 重点采样
                    head_imp = []
                    for i in temp_head:
                        head_imp.append(node_important[i])
                    # 将对应的前n个索引提取出来
                    l = zip(head_imp, temp_head, temp_relation, temp_tail)
                    l = sorted(l)
                    # 'unzip'
                    try:
                        head_imp_sorted, temp_head_sorted, temp_relation_sorted, temp_tail_sorted = zip(*l)
                    except BaseException as e:
                        continue
                    memories_h.extend(temp_head_sorted[0:args.n_memory])
                    memories_r.extend(temp_relation_sorted[0:args.n_memory])
                    memories_t.extend(temp_tail_sorted[0:args.n_memory])

                # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                # this won't happen for h = 0, because only the items that appear in the KG have been selected
                # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
                try:
                    if len(memories_h) == 0:
                        ripple_set[user].append(ripple_set[user][-1])
                    else:
                        # sample a fixed-size 1-hop memory for each user
                        replace = len(memories_h) < args.n_memory  # 如果超过采样数量则使用重点采样
                        indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                        memories_h = [memories_h[i] for i in indices]
                        memories_r = [memories_r[i] for i in indices]
                        memories_t = [memories_t[i] for i in indices]
                        ripple_set[user].append((memories_h, memories_r, memories_t))
                except:
                    pass

    print(f'ripple set 有{len(ripple_set)}个数据')
    print(f'保存ripple set数据')
    np.save('./data/ripple_set.npy', ripple_set)
    # with open('./data/ripple_set.json', 'w', encoding='utf8')as fp:
    #     json.dump(ripple_set, fp, ensure_ascii=False)
    return ripple_set
