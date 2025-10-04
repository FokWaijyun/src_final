import argparse
import os

from tqdm import tqdm
import json

import pandas as pd
import numpy as np

entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'brick_data': 'ratings.dat',
                         'component': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::',
            'brick_data': '::',
            'component': '::',
            'book': ';',
            'news': '\t'})
THRESHOLD = dict({'movie': 4,
                  'brick_data': 1,
                  'component': 1,
                  'book': 0,
                  'news': 0})


def read_item_index_to_entity_id_file(args):
    file = '../data/' + args.dataset + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        if args.dataset != 'movie':
            entity_id2index[item_index] = i
        else:
            entity_id2index[satori_id] = i
        i += 1


def convert_rating(args):
    file = '../data/' + args.dataset + '/' + RATING_FILE_NAME[args.dataset]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[args.dataset])

        # remove prefix and suffix quotation marks for BX dataset
        if args.dataset == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = array[0]

        rating = float(array[2])
        # 这里会对数据产生去重效果
        if rating >= THRESHOLD[args.dataset]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print(f'具有用户行为的用户数量:{len(user_pos_ratings)}')
    print('converting rating file ...')
    writer = open('../data/' + args.dataset + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    # 生成决策树训练数据
    ratings_pd = pd.DataFrame(columns=['user_id', 'build_id', 'interaction'])
    id2item = {id: build for build, id in item_index_old2new.items()}
    for user_index_old, pos_item_set in tqdm(user_pos_ratings.items()):
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
            ratings_pd.at[ratings_pd.shape[0]] = [user_index_old, id2item[item], 1]
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set) * args.neg_rate,
                                     replace=True):  # 控制负样本比例
            writer.write('%d\t%d\t0\n' % (user_index, item))
            ratings_pd.at[ratings_pd.shape[0]] = [user_index_old, id2item[item], 0]
    writer.close()
    ratings_pd.to_csv('ratings.csv', index=False, header=False, encoding='utf-8')
    # 保存用户id索引:用户切分数据时使用
    with open(f'../data/{args.dataset}/map/user2id.json', 'w', encoding='utf8') as fp:
        json.dump(user_index_old2new, fp, ensure_ascii=False)
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg(args):
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + args.dataset + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if args.dataset == 'movie':
        files.append(open('../data/' + args.dataset + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + args.dataset + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + args.dataset + '/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            try:
                head_old = array[0]
                relation_old = array[1]
                tail_old = array[2]
            except:
                print('实体出错:', line)

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def preprocess():
    np.random.seed(555)
    print(os.getcwd())

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    parser.add_argument('-d', '--dataset', type=str, default='brick_data', help='which dataset to preprocess')
    parser.add_argument('--neg_rate', type=int, default=1, help='负样本比例')
    # parser.add_argument('-d', '--dataset', type=str, default='component', help='which dataset to preprocess')
    # parser.add_argument('-d', '--dataset', type=str, default='book', help='which dataset to preprocess')
    args = parser.parse_args()
    args.dataset = args.dataset

    read_item_index_to_entity_id_file(args)
    convert_rating(args)
    convert_kg(args)
    with open(f'../data/{args.dataset}/map/entity_id2index.json', 'w', encoding='utf8') as fp:
        json.dump(entity_id2index, fp, ensure_ascii=False)
    with open(f'../data/{args.dataset}/map/relation_id2index.json', 'w', encoding='utf8') as fp:
        json.dump(relation_id2index, fp, ensure_ascii=False)
    with open(f'../data/{args.dataset}/map/item_index_old2new.json', 'w', encoding='utf8') as fp:
        json.dump(item_index_old2new, fp, ensure_ascii=False)
    print('done')
    return


if __name__ == '__main__':
    preprocess()
