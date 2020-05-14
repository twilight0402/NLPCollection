import os
import logging
import jieba

import torch
import torch.nn as nn
from torchtext.data import Field, LabelField, TabularDataset
from classification.configure import Config

logger = logging.getLogger(__name__)


def build_and_cache_dataset(config: Config, mode='train'):
    """
    返回每个属性的Field，以及所有的属性的值
    (id, category, news), datasets
    (Field, Field, Field), TabularDataset
    """
    # id 已经序列化
    ID = Field(sequential=False, use_vocab=False)
    CATEGORY = LabelField(sequential=False, use_vocab=True, is_target=True)
    NEWS = Field(sequential=True, tokenize=jieba.lcut, include_lengths=True,)

    fields = [
        ('id', ID),
        (None, None),
        ('category', CATEGORY),
        ('news', NEWS),
    ]

    logger.info("从当前目录创建特征 %s", config.dataset_dir)

    # `\t` 分割
    dataset = TabularDataset(
        os.path.join(config.dataset_dir, f'{mode}.csv'),
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '\t'},
    )

    # TabularDataset.split()
    features = ((ID, CATEGORY, NEWS), dataset)
    return features


def save_model(config: Config, model, optimizer=None, scheduler=None, global_step=0):
    '''
    保存模型
    :param config:
    :param model:
    :param optimizer:
    :param scheduler:
    :param global_step:
    :return:
    '''
    # Save model checkpoint
    output_dir = os.path.join(config.model_save_path, "ckpt-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take care of distributed/parallel training
    logger.info("模型已保存： %s", output_dir)
    logger.info("保存optimizer 和scheduler： %s", output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
