# from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os
from gensim import corpora
import random

PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path="../datasets/", tokenizer=None, pad_size=32, type="train"):
    """
    返回结果 4个list ids, lable, ids_len, mask
    list(list(padsize), int, int, list(padsize))

    :param file_path:
    :param seq_len:
    :param pad_size:
    :return: list((list(id), int(label), int(len), list(mask)))
    """
    contents = []
    labels = []
    with open(os.path.join(file_path, type+".csv"), 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.split('\t')
            if len(line) == 4:
                _, _, label, content = line
            else:
                _, _, label, content, tail = line
                content = content + tail

            labels.append(label)
            token = tokenizer.tokenize(content)    # 切成字列表，list(str)
            token = [CLS] + token
            seq_len = len(token)        # 真实的句子长度
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)   # 转换成对应的id列表， list(int)

            if pad_size:
                # 句子太短
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                # 句子太长
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size      # 切断句子的长度，最大32
            contents.append((token_ids, label, seq_len, mask))

        # 针对id建立词典
        dictory = corpora.Dictionary([labels])
        # 将label 转换为id
        contents = [(token_ids, dictory.token2id[label], seq_len, mask) for (token_ids, label, seq_len, mask) in contents]
        random.shuffle(contents)
        print("已经打乱数据: ", type)
    return contents


def bulid_dataset(dataset_path: str, tokenizer, pad_size):
    """
    dataset_path: 当前目录下有train.csv文件
    """
    datasetpkl = dataset_path + "/dataset.pkl"
    if os.path.exists(datasetpkl):
        dataset = pkl.load(open(datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(dataset_path, tokenizer, pad_size, "train")
        dev = load_dataset(dataset_path, tokenizer, pad_size, "dev")
        test = load_dataset(dataset_path, tokenizer, pad_size, "test")

        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(datasetpkl, 'wb'))
    return train, dev, test


class DatasetIterator(object):
    """
    迭代器对象
    返回4个张量 ： (x, seq_len, mask), y  ==> (list(list(int)), list(int), list(list(int)), int)
    - 实现了__iter__函数对象是 可迭代对象Iteratable，他应该返回一个实现了__next__的对象
    - 同时实现了__iter__和__next__的是迭代器(Iterator)
    - __next__函数中通过抛出一个StopIteration异常表示迭代结束
    """
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False            # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """
        datas (token_ids, label, seq_len, mask)
        """
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)   # x样本数据 list(list(id))
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)   # 标签数据 int(label)

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)     # 每一个序列的真实长度 int(len)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)        # list(list(int))

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def bulid_iterator(dataset, batch_size, device):
    '''
    dataset: 已经获得到的数据集 (token_ids, label, seq_len, mask)
    '''
    iter = DatasetIterator(dataset, batch_size, device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
