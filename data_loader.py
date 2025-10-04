import collections
import networkx as nx
from tqdm import tqdm
from collections import Counter
import json
import os
import numpy as np
import pandas as pd


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    if args.important_sample == 'close':
        print('Close important sample mode')
        ripple_set = close_important_get_ripple_set(args, kg, user_history_dict)
    elif args.important_sample == 'pagerank':
        print('Pagerank important sample mode')
    elif args.important_sample == 'eig':
        print('eig important sample mode')
        ripple_set = imp_get_ripple_set(args, kg, user_history_dict)
    elif args.important_sample == 'betweenness':
        print('betweenness important sample mode')
        ripple_set = imp_get_ripple_set(args, kg, user_history_dict)
    elif args.important_sample == 'info':
        print('info important sample mode')
        ripple_set = imp_get_ripple_set(args, kg, user_history_dict)
    else:
        ripple_set = get_ripple_set(args, kg, user_history_dict)

    # 保存数据给对比模型
    pd.DataFrame(train_data).to_csv("./cache/train.csv", index=False, header=False)
    pd.DataFrame(eval_data).to_csv("./cache/eval.csv", index=False, header=False)
    pd.DataFrame(test_data).to_csv("./cache/test.csv", index=False, header=False)
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


def dataset_split(args, rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    # eval_ratio = 0.2
    # test_ratio = 0.0
    n_ratings = rating_np.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    user_indices = range(n_ratings)

    # traverse training map, only keeping the users with positive ratings
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

    # 打印相关信息
    print(f'数据集数量: train:{len(train_indices)},test:{len(eval_indices)}, test:{len(test_indices)}')
    # print(f'正负样本比例:\n'
    #       f'train:{Counter(rating_np[train_indices][2])}\n'
    #       f'eval:{Counter(rating_np[eval_indices][2])}\n'
    #       f'test:{Counter(rating_np[test_indices][2])}\n')

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    # np.save(f'test_{args.date}.npy', test_data)

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

    kg = construct_kg(args, kg_np)

    return n_entity, n_relation, kg


def construct_kg(args, kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))

    # np.save(f'kg_{args.date}.npy', kg)
    return kg


def get_ripple_set(args, kg, user_history_dict):
    if os.path.isfile(f'ripple_set_{args.date}.npy'):
        print('读取ripple set文件')
        ripple_set = np.load(f'ripple_set_{args.date}.npy', allow_pickle=True)
        ripple_set = ripple_set[()]
        return ripple_set

    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
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

    print(f'ripple set 有{len(ripple_set)}个数据')
    # 保存ripple set
    # np.save(f'ripple_set_{args.date}.npy', ripple_set)

    return ripple_set


def pagerank():
    kg_final = pd.read_csv(f'../data/brick_data/kg_final.txt', sep='\t', encoding='utf_8_sig', header=None,
                           names=['head', 'relation', 'tail', 'time'])
    del kg_final['relation']
    del kg_final['time']
    kg_final.to_csv(f'./data/kg_edge_list.txt', sep='\t', header=None, index=False)

    G = nx.read_edgelist(f'./data/kg_edge_list.txt', create_using=nx.Graph(), nodetype=int,
                         data=(('weight', float),))
    imp = nx.pagerank(G, alpha=0.85)
    node_imp = {int(key): val for key, val in imp.items()}
    with open('./data/node_important.json', 'w', encoding='utf8') as fp:
        json.dump(node_imp, fp, ensure_ascii=False, sort_keys=True)

    return node_imp


def get_close_imp():
    kg_final = pd.read_csv(f'../data/brick_data/kg_final.txt', sep='\t', encoding='utf_8_sig', header=None,
                           names=['head', 'relation', 'tail', 'time'])
    del kg_final['relation']
    del kg_final['time']
    kg_final.to_csv(f'./data/kg_edge_list.txt', sep='\t', header=None, index=False)

    G = nx.read_edgelist(f'./data/kg_edge_list.txt', create_using=nx.Graph(), nodetype=int,
                         data=(('weight', float),))
    imp = nx.closeness_centrality(G)
    node_imp = {int(key): val for key, val in imp.items()}
    with open('./data/node_close_important.json', 'w', encoding='utf8') as fp:
        json.dump(node_imp, fp, ensure_ascii=False, sort_keys=True)

    return node_imp


def get_eig_imp():
    kg_final = pd.read_csv(f'../data/brick_data/kg_final.txt', sep='\t', encoding='utf_8_sig', header=None,
                           names=['head', 'relation', 'tail', 'time'])
    del kg_final['relation']
    del kg_final['time']
    kg_final.to_csv(f'./data/kg_edge_list.txt', sep='\t', header=None, index=False)

    G = nx.read_edgelist(f'./data/kg_edge_list.txt', create_using=nx.Graph(), nodetype=int,
                         data=(('weight', float),))
    eig = nx.eigenvector_centrality(G, max_iter=500)
    imp = dict(sorted(eig.items(), key=lambda x: x[1], reverse=True))
    node_imp = {int(key): val for key, val in imp.items()}
    with open('./data/node_close_important.json', 'w', encoding='utf8') as fp:
        json.dump(node_imp, fp, ensure_ascii=False, sort_keys=True)

    return node_imp


def important_get_ripple_set(args, kg, user_history_dict):
    if os.path.isfile(f'ripple_set_sampling_{args.date}.npy'):
        print('读取ripple set重点次啊样文件文件')
        ripple_set = np.load(f'ripple_set_{args.date}.npy', allow_pickle=True)
        ripple_set = ripple_set[()]
        return ripple_set

    print('constructing ripple set ...')
    node_important = pagerank()
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            temp_head = []
            temp_relation = []
            temp_tail = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    temp_head.append(entity)
                    temp_tail.append(tail_and_relation[0])
                    temp_relation.append(tail_and_relation[1])

                    # memories_h.append(entity)
                    # memories_r.append(tail_and_relation[1])
                    # memories_t.append(tail_and_relation[0])

            # 重点采样
            head_imp = []
            for i in temp_head:
                head_imp.append(node_important[i])
            # 将对应的前n个索引提取出来
            index = np.argsort(-np.array(head_imp))
            temp_head_sorted = [temp_head[i] for i in index]
            temp_relation_sorted = [temp_relation[i] for i in index]
            temp_tail_sorted = [temp_tail[i] for i in index]

            # 截断比例重点采样逻辑-防止采样后之后一种节点
            ## 计算有多少不同的节点，取前n_memory个不同节点进行比例抽取
            # step0: 采样前n个不同节点
            count = 0
            nodes = []
            for i in temp_head_sorted:
                if count >= args.n_memory:
                    break
                if i not in nodes:
                    nodes.append(i)
                    count += 1

            # step1: 计算前n个节点重要性比例
            import math
            nodes_imp = [node_important[i] for i in nodes]
            total = sum(nodes_imp)
            nodes_imp_rate = [i / total * args.n_memory for i in nodes_imp]
            nodes_imp_rate_approximate = [math.ceil(i / total * args.n_memory) for i in nodes_imp]

            # setp3: 如果超过，逆序逐个减1
            # for i in range(sum(nodes_imp_rate_approximate) - args.n_memory):
            #     nodes_imp_rate_approximate[-(i + 1)] -= 1

            node2num = {}
            for index, head in enumerate(nodes):
                if nodes_imp_rate_approximate[index] <= 0:
                    break
                else:
                    node2num[head] = nodes_imp_rate_approximate[index]

            # step4: 获取头节点对应的知识集，随机抽取对应个数知识。
            # 少于比例则重复抽取
            h = []
            r = []
            t = []
            for head, num in node2num.items():
                replace = len(kg[head]) < num  # 如果超过采样数量则使用重点采样
                head_knowledge_set = kg[head]
                indices = np.random.choice(len(kg[head]), size=num, replace=replace)
                for i in indices:
                    h.append(head)
                    r.append(head_knowledge_set[i][1])
                    t.append(head_knowledge_set[i][0])  # 尾节点

            # final
            # memories_h.extend(temp_head_sorted[0:args.n_memory])
            # memories_r.extend(temp_relation_sorted[0:args.n_memory])
            # memories_t.extend(temp_tail_sorted[0:args.n_memory])
            memories_h = h
            memories_r = r
            memories_t = t

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
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

    print(f'ripple set 有{len(ripple_set)}个数据')
    # 保存ripple set
    # np.save(f'ripple_set_{args.date}.npy', ripple_set)
    np.save(f'ripple_set_sampling_{args.date}.npy', ripple_set)

    return ripple_set


def close_important_get_ripple_set(args, kg, user_history_dict):
    """
    紧密中心性重要性采样
    """
    if os.path.isfile(f'ripple_set_close_sampling_{args.date}.npy'):
        print('读取ripple set重点次啊样文件文件')
        ripple_set = np.load(f'ripple_set_close_sampling_{args.date}.npy', allow_pickle=True)
        ripple_set = ripple_set[()]
        return ripple_set

    print('constructing ripple set ...')
    node_important = get_close_imp()
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            temp_head = []
            temp_relation = []
            temp_tail = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    temp_head.append(entity)
                    temp_tail.append(tail_and_relation[0])
                    temp_relation.append(tail_and_relation[1])

                    # memories_h.append(entity)
                    # memories_r.append(tail_and_relation[1])
                    # memories_t.append(tail_and_relation[0])

            # 重点采样
            head_imp = []
            for i in temp_head:
                head_imp.append(node_important[i])
            # 将对应的前n个索引提取出来
            index = np.argsort(-np.array(head_imp))
            temp_head_sorted = [temp_head[i] for i in index]
            temp_relation_sorted = [temp_relation[i] for i in index]
            temp_tail_sorted = [temp_tail[i] for i in index]

            # 截断比例重点采样逻辑-防止采样后之后一种节点
            ## 计算有多少不同的节点，取前n_memory个不同节点进行比例抽取
            # step0: 采样前n个不同节点
            count = 0
            nodes = []
            for i in temp_head_sorted:
                if count >= args.n_memory:
                    break
                if i not in nodes:
                    nodes.append(i)
                    count += 1

            # step1: 计算前n个节点重要性比例
            import math
            nodes_imp = [node_important[i] for i in nodes]
            total = sum(nodes_imp)
            nodes_imp_rate = [i / total * args.n_memory for i in nodes_imp]
            nodes_imp_rate_approximate = [math.ceil(i / total * args.n_memory) for i in nodes_imp]

            # setp3: 如果超过，逆序逐个减1
            # for i in range(sum(nodes_imp_rate_approximate) - args.n_memory):
            #     nodes_imp_rate_approximate[-(i + 1)] -= 1

            node2num = {}
            for index, head in enumerate(nodes):
                if nodes_imp_rate_approximate[index] <= 0:
                    break
                else:
                    node2num[head] = nodes_imp_rate_approximate[index]

            # step4: 获取头节点对应的知识集，随机抽取对应个数知识。
            # 少于比例则重复抽取
            h = []
            r = []
            t = []
            for head, num in node2num.items():
                replace = len(kg[head]) < num  # 如果超过采样数量则使用重点采样
                head_knowledge_set = kg[head]
                indices = np.random.choice(len(kg[head]), size=num, replace=replace)
                for i in indices:
                    h.append(head)
                    r.append(head_knowledge_set[i][1])
                    t.append(head_knowledge_set[i][0])  # 尾节点

            # final
            # memories_h.extend(temp_head_sorted[0:args.n_memory])
            # memories_r.extend(temp_relation_sorted[0:args.n_memory])
            # memories_t.extend(temp_tail_sorted[0:args.n_memory])
            memories_h = h
            memories_r = r
            memories_t = t

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
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

    print(f'ripple set 有{len(ripple_set)}个数据')
    # 保存ripple set
    # np.save(f'ripple_set_{args.date}.npy', ripple_set)
    np.save(f'ripple_set_close_sampling_{args.date}.npy', ripple_set)

    return ripple_set


def imp_get_ripple_set(args, kg, user_history_dict):
    """
    使用重要性性采样获取涟漪集合
    可以更换采样器
    """
    if os.path.isfile(f'ripple_set_{args.important_sample}_sampling_{args.date}.npy'):
        print('读取ripple set文件')
        ripple_set = np.load(f'ripple_set_{args.important_sample}_sampling_{args.date}.npy', allow_pickle=True)
        ripple_set = ripple_set[()]
        return ripple_set

    print('constructing ripple set ...')

    kg_final = pd.read_csv(f'../data/brick_data/kg_final.txt', sep='\t', encoding='utf_8_sig', header=None,
                           names=['head', 'relation', 'tail', 'time'])
    del kg_final['relation']
    del kg_final['time']
    kg_final.to_csv(f'./data/kg_edge_list.txt', sep='\t', header=None, index=False)

    G = nx.read_edgelist(f'./data/kg_edge_list.txt', create_using=nx.Graph(), nodetype=int,
                         data=(('weight', float),))
    # 选择采样器
    if args.important_sample == 'eig':
        eig = nx.eigenvector_centrality(G, max_iter=500)
        imp = dict(sorted(eig.items(), key=lambda x: x[1], reverse=True))
    elif args.important_sample == 'betweenness':
        betweenness = nx.betweenness_centrality(G)
        imp = dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True))
    elif args.important_sample == 'info':
        current_flow_closeness_centrality = nx.current_flow_closeness_centrality(G)
        imp = dict(sorted(current_flow_closeness_centrality.items(), key=lambda x: x[1], reverse=False))
    node_important = {int(key): val for key, val in imp.items()}

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            temp_head = []
            temp_relation = []
            temp_tail = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    temp_head.append(entity)
                    temp_tail.append(tail_and_relation[0])
                    temp_relation.append(tail_and_relation[1])

                    # memories_h.append(entity)
                    # memories_r.append(tail_and_relation[1])
                    # memories_t.append(tail_and_relation[0])

            # 重点采样
            head_imp = []
            for i in temp_head:
                head_imp.append(node_important[i])
            # 将对应的前n个索引提取出来
            index = np.argsort(-np.array(head_imp))
            temp_head_sorted = [temp_head[i] for i in index]
            # temp_relation_sorted = [temp_relation[i] for i in index]
            # temp_tail_sorted = [temp_tail[i] for i in index]

            # 截断比例重点采样逻辑-防止采样后之后一种节点
            ## 计算有多少不同的节点，取前n_memory个不同节点进行比例抽取
            # step0: 采样前n个不同节点
            count = 0
            nodes = []
            for i in temp_head_sorted:
                if count >= args.n_memory:
                    break
                if i not in nodes:
                    nodes.append(i)
                    count += 1

            # step1: 计算前n个节点重要性比例
            import math
            nodes_imp = [node_important[i] for i in nodes]
            total = sum(nodes_imp)
            # nodes_imp_rate = [i / total * args.n_memory for i in nodes_imp]
            nodes_imp_rate_approximate = [math.ceil(i / total * args.n_memory) for i in nodes_imp]

            # setp3: 如果超过，逆序逐个减1
            # for i in range(sum(nodes_imp_rate_approximate) - args.n_memory):
            #     nodes_imp_rate_approximate[-(i + 1)] -= 1

            node2num = {}
            for index, head in enumerate(nodes):
                if nodes_imp_rate_approximate[index] <= 0:
                    break
                else:
                    node2num[head] = nodes_imp_rate_approximate[index]

            # step4: 获取头节点对应的知识集，随机抽取对应个数知识。
            # 少于比例则重复抽取
            h = []
            r = []
            t = []
            for head, num in node2num.items():
                replace = len(kg[head]) < num  # 如果超过采样数量则使用重点采样
                head_knowledge_set = kg[head]
                indices = np.random.choice(len(kg[head]), size=num, replace=replace)
                for i in indices:
                    h.append(head)
                    r.append(head_knowledge_set[i][1])
                    t.append(head_knowledge_set[i][0])  # 尾节点

            # final
            # memories_h.extend(temp_head_sorted[0:args.n_memory])
            # memories_r.extend(temp_relation_sorted[0:args.n_memory])
            # memories_t.extend(temp_tail_sorted[0:args.n_memory])
            memories_h = h
            memories_r = r
            memories_t = t

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
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

    print(f'ripple set 有{len(ripple_set)}个数据')
    # 保存ripple set
    # np.save(f'ripple_set_{args.date}.npy', ripple_set)
    np.save(f'ripple_set_{args.important_sample}_sampling_{args.date}.npy', ripple_set)

    return ripple_set


if __name__ == '__main__':
    print(os.getcwd())
    pagerank()
