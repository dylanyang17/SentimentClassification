import os
import time

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from config import Config
from cnn import CNN
import re


def load_word2vec(path):
    print('Loading word2vec...')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        word_num, vec_len = list(map(int, lines[0].split(' ')))
        print(word_num, vec_len)
        word2vec = {}
        for i in range(1, word_num + 1):
            if i % 5000 == 0:
                print('%d / %d' % (i, word_num))
            items = lines[i].strip().split(' ')
            vec = torch.tensor(list(map(float, items[1:vec_len+1])))
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

                label_item = torch.zeros([label_len], dtype=torch.float)
                sen = tmp[1].split(':')
                for j in range(2, label_len+2):
                    label_item[j-2] = float(re.match('([0-9]*).*', sen[j]).group(1))
                label_item /= label_item.sum()
                self.labels.append(label_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.text_lengths[index], self.labels[index]


def calc_acc(net, device, dataloaders):
    ret = []
    with torch.no_grad():
        for dataloader in dataloaders:
            correct = torch.zeros([1]).to(device)
            total = torch.zeros([1]).to(device)
            for i, data in enumerate(dataloader):
                inputs, _, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                prediction = torch.argmax(outputs, 1)
                correct += (prediction == torch.argmax(labels, 1)).sum().float()
                total += len(labels)
            ret.append(float(correct/total))
    return ret


def train(net, device, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    now = time.clock()
    for epoch in range(Config.epoch_num):
        running_loss = 0.0
        cnt = 0
        for i, data in enumerate(train_dataloader):
            inputs, _, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            cnt += 1

        delta = time.clock() - now
        now = time.clock()
        train_acc, valid_acc, test_acc = calc_acc(net, device, [train_dataloader, valid_dataloader, test_dataloader])
        print('[%d] loss: %.3f  cost: %.3f  train_acc: %.3f  valid_acc: %.3f  test_acc: %.3f' %
              (epoch + 1, running_loss / cnt, delta, train_acc, valid_acc, test_acc))


def train_cnn(train_dataloader, valid_dataloader, test_dataloader):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = CNN().to(device)
    optimizer = optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum)
    criterion = nn.MSELoss()
    train(net, device, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader)


def train_rnn():
    pass
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # net = RNN().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum)
    # criterion = nn.MSELoss()
    # train(net, device, optimizer, criterion, train_dataloader, test_dataloader)


if __name__ == '__main__':
    word2vec = load_word2vec(Config.reduced_vec_path)
    train_data = MyDataset(Config.train_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    valid_data = MyDataset(Config.valid_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    test_data = MyDataset(Config.test_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)
    valid_dataloader = DataLoader(dataset=valid_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)
    test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)

    train_cnn(train_dataloader, valid_dataloader, test_dataloader)
