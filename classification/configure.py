import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import os


class Config(object):
    """
    配置参数， 所有参数
    """
    def __init__(self, dataset="/datasets/classification"):
        self.model_name = 'Bert'
        self.num_epochs = 20        # epoch数
        self.batch_size = 256       # batch_size
        self.pad_size = 32          # 每句话处理的长度(短填，长切）
        self.learning_rate = 1e-5   # 学习率
        self.hidden_size = 768      # bert隐层层个数, 要和预训练数据一致
        self.class_list = None      # 类别列表
        self.num_classes = 5        # 类别数

        # 相关目录
        self.model_save_path = dataset + '/saved_model' + self.model_name + '.ckpt'
        self.dataset_dir = dataset
        self.embedding_dir = dataset + "/embedding"
        self.bert_pretrain_path = dataset + "/bert_pretrain"  # bert预训练模型位置

        # bert切词器（需要与训练数据）
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrain_path)
        self.require_improvement = 1000    # 若超过 1000bacth 效果还没有提升，提前结束训练

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






