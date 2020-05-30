import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from config import Config

def get_reduced_vec():
    """
    简化词向量并写入 reduced.sgns.sogounews.bigram-char
    """
    # 读入 word2vec
    with open(Config.vec_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        word_num, vec_len = list(map(int, lines[0].split(' ')))
        print(word_num, vec_len)
        word2vec = {}
        for i in range(1, word_num + 1):
            if i % 500 == 0:
                print(i)
            items = lines[i].strip().split(' ')
            vec = torch.zeros([vec_len])
            if len(items) < vec_len + 1:
                print('continue')
                continue
            for j in range(1, vec_len + 1):
                vec[j - 1] = float(items[j])
            word2vec[items[0]] = vec

    # 标记所有出现过的词
    has = {}
    with open(Config.train_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split('\t')[2].split(' ')
            for word in words:
                has[word] = True
    with open(Config.test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split('\t')[2].split(' ')
            for word in words:
                has[word] = True

    # 写入 reduced
    with open(os.path.join(Config.data_root, 'reduced.sgns.sogounews.bigram-char'), 'w', encoding='utf-8') as f:
        cnt = 0
        for k, v in word2vec.items():
            if has.get(k) is not None:
                cnt += 1
        f.write('%d %d\n' % (cnt, vec_len))
        print(cnt, vec_len)
        ttt = 0
        for k, v in word2vec.items():
            ttt += 1
            if ttt % 1000 == 0:
                print(ttt)
            if has.get(k) is not None:
                f.write('%s ' % k)
                for i in range(vec_len):
                    f.write('%.6f%s' % (float(v[i]), ' ' if i != vec_len - 1 else '\n'))
