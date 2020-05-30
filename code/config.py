import os
import torch


class Config:
    data_root = '../data'
    vec_path = os.path.join(data_root, 'sgns.sogounews.bigram-char')
    reduced_vec_path = os.path.join(data_root, 'reduced.sgns.sogounews.bigram-char')
    train_data_path = os.path.join(data_root, 'sinanews.train')
    test_data_path = os.path.join(data_root, 'sinanews.test')
    label_len = 8
    vec_len = 300
    seq_len = 300
    unk_vec = torch.rand([vec_len])*2 - 1
    train_batch_size = 32
