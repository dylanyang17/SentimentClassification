import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from config import Config
import re


def load_word2vec(path):
    print('Loading word2vec...')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        word_num, vec_len = list(map(int, lines[0].split(' ')))
        print(word_num, vec_len)
        word2vec = {}
        for i in range(1, word_num + 1):
            if i % 500 == 0:
                print('%d / %d' % (i, word_num))
            items = lines[i].strip().split(' ')
            vec = torch.tensor(map(float, items[1:vec_len+1]))
            word2vec[items[0]] = vec
    print('Loaded word2vec successfully.')
    return word2vec


class MyDataset(Dataset):
    """
    self.data 为一个 list，第 i 个元素即第 i 个数据
    self.labels 为一个 list，第 i 个元素即第 i 个 label
    self.seq_len 为每个文本的长度，超过则截断
    self.vec_len 为每个词向量的长度
    每个数据维度为 [seq_len, vec_len]，每个 label 维度为 [label_len]
    """
    def __init__(self, data_path, seq_len, vec_len, label_len, word2vec):
        self.data = []
        self.labels = []
        self.text_lengths = []
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip('\n ').split('\n')
            for i, line in enumerate(lines):
                tmp = line.split('\t')
                words = tmp[2].strip().split(' ')
                min_len = min(seq_len, len(words))
                self.text_lengths.append(min_len)
                data_item = torch.zeros([seq_len, vec_len])
                for j in range(min_len):
                    vec = word2vec.get(words[j])
                    if vec is None:
                        vec = Config.unk_vec
                    data_item[j] = vec
                self.data.append(data_item)

                label_item = torch.zeros([label_len])
                sen = tmp[1].split(':')
                for j in range(1, label_len+1):
                    label_item[j-1] = float(re.match('([0-9]*).*', sen[j]).group(1))
                label_item /= label_item.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


if __name__ == '__main__':
    word2vec = load_word2vec(Config.reduced_vec_path)
    train_data = MyDataset(Config.train_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
