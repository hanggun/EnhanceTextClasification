import os
import numpy as np
from glob import glob
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def process_agnews():
    dir = '/home/zxa/ps/open_data/classification/AGNews/'
    train_name = 'train.csv'

    D = []
    with open(dir+train_name, 'r', encoding='utf8') as f:
        for line in f:
            D.append(line)
    np.random.shuffle(D)

    train_num = int(len(D)*0.9)
    train_data = D[:train_num]
    dev_data = D[train_num:len(D)]

    with open(dir+'train.csv', 'w', encoding='utf8') as f:
        f.write(''.join(train_data))
    with open(dir+'dev.csv', 'w', encoding='utf8') as f:
        f.write(''.join(dev_data))


def process_imdb():
    filedir = '/home/zxa/ps/open_data/classification/aclImdb/'
    classes = ['pos', 'neg']
    modes = ['train', 'test']
    train_data = []
    test_data = []
    for m in modes:
        for c in classes:
            files = glob(os.path.join(filedir, m, c, '*'))
            for file in tqdm(files):
                with open(file, encoding='utf8') as f:
                    text = f.read().strip()
                    text = re.sub('<br />', '', text)
                    if m == 'train':
                        if c == 'pos':
                            train_data.append('1|||'+text)
                        else:
                            train_data.append('0|||' + text)
                    else:
                        if c == 'pos':
                            test_data.append('1|||'+text)
                        else:
                            test_data.append('0|||' + text)
    train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)
    with open(os.path.join(filedir, 'train.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(train_data))
    with open(os.path.join(filedir, 'dev.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(dev_data))
    with open(os.path.join(filedir, 'test.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(test_data))


def process_bertGCNdata():
    datasets = ['mr', '20ng', 'R8', 'R52', 'ohsumed']
    for dataset in datasets:
        sample_lens = []
        categories = set()
        doc_name_list = []
        doc_train_list = []
        doc_test_list = []
        f = open('/home/zxa/ps/open_data/classification/BertGCNdata/' + dataset + '.txt', 'r')
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
        f.close()

        doc_content_list = []
        f = open('/home/zxa/ps/open_data/classification/BertGCNdata/corpus/' + dataset + '.clean.txt', 'r')
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
            sample_lens.append(len(line.strip().split()))
        f.close()

        train_ids = []
        for train_name in doc_train_list:
            train_id = doc_name_list.index(train_name)
            train_ids.append(train_id)
        # print(train_ids)
        random.shuffle(train_ids)
        train_ids = train_ids[:int(len(train_ids)*0.9)]

        train_data = []
        for id in train_ids:
            label = doc_name_list[int(id)].split('\t')[-1]
            text = doc_content_list[int(id)]
            train_data.append(label+'|||'+text)
            categories.add(label)

        with open(f'/home/zxa/ps/open_data/classification/{dataset}/'+'train.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(train_data))
        print('data saved in '+f'/home/zxa/ps/open_data/classification/{dataset}/'+'train.txt')

        test_ids = []
        for test_name in doc_test_list:
            test_id = doc_name_list.index(test_name)
            test_ids.append(test_id)
        # print(test_ids)
        random.shuffle(test_ids)

        test_data = []
        for id in test_ids:
            label = doc_name_list[int(id)].split('\t')[-1]
            text = doc_content_list[int(id)]
            test_data.append(label + '|||' + text)
            categories.add(label)

        with open(f'/home/zxa/ps/open_data/classification/{dataset}/'+'test.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(test_data))
        print('data saved in ' + f'/home/zxa/ps/open_data/classification/{dataset}/' + 'test.txt')

        print('label length', len(categories))
        sample_lens.sort()
        print(f"{dataset}: maxlen {max(sample_lens)}, avglen {np.mean(sample_lens)}, minlen {min(sample_lens)} "
              f".75 {sample_lens[int(len(sample_lens)*0.75)+1]} .85 {sample_lens[int(len(sample_lens)*0.85)+1]}")
        print(categories)


if __name__ == '__main__':
    # process_agnews()
    # process_imdb()
    process_bertGCNdata()