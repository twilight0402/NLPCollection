import os
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support
from classification.models.Bert import TextClassifier
from classification.configure import Config
from classification import utils
from pytorch_pretrained_bert import BertAdam
import torch.nn.functional as F


def train(config):

    #### 1. 数据集
    tokenizer = config.tokenizer
    # (token_ids, label, seq_len, mask)
    train_data, dev_data, test_data = utils.bulid_dataset(
        dataset_path=config.dataset_dir,
        tokenizer=tokenizer,
        pad_size=config.pad_size)

    # 返回的每一条数据格式： (x, seq_len, mask), y
    train_iter = utils.bulid_iterator(train_data, batch_size=config.batch_size, device=config.device)
    dev_iter = utils.bulid_iterator(dev_data, batch_size=config.batch_size, device=config.device)
    test_iter = utils.bulid_iterator(test_data, batch_size=config.batch_size, device=config.device)

    ### 2. 模型
    model = TextClassifier(
        output_dim=5,       # 输出的维度， 默认5
        dropout=config.dropout,
        bert_pretrain_path=config.bert_pretrain_dir,
        bertConfig=config
    )

    # optimizer, lr_scheduler, criterion
    model.to(config.device)

    # 拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=bertConfig.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_train_epochs)

    global_step = 0
    model.zero_grad()

    for i in range(config.num_epochs):
        # epoch_iterator = tqdm(train_iter
        for step, batch in enumerate(train_iter):
            model.train()

            # 返回的每一条数据格式： (x, seq_len, mask), y
            x, category = batch
            preds = model(x)
            model.zero_grad()

            loss = F.cross_entropy(preds, category)
            loss.backward()

            # NOTE: Update model, optimizer should update before scheduler
            optimizer.step()
            # scheduler.step()
            global_step += 1

            # # NOTE: save model
            # if args.save_steps > 0 and global_step % args.save_steps == 0:
            #     save_model(args, model, optimizer, scheduler, global_step)

            if step % 50 == 0:
                print(f"{step}/{i+1}: loss {loss.cpu().data.numpy()}")

            if step % 300 == 0:
                evaluate(model, dev_iter)

            # evaluate(model, dev_iter)

        # 动态修改参数学习率
        optimizer.param_groups[0]['lr'] *= 0.8  # 每过5轮，衰减一次学习率
        print(optimizer.param_groups[0]['lr'])
    evaluate(model, dev_iter)


def evaluate(model, iter):
    # NOTE: Eval!
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss, eval_steps = 0.0, 0
    labels_list, preds_list = [], []

    # TODO CHECK
    # index = 1
    for batch in iter:
        x, y = batch
        with torch.no_grad():
            # print(index)
            # if index == 157 :
            #     pass
            #     print("aaa")
            # index += 1
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            eval_loss += loss.item()

        eval_steps += 1
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds)
        labels_list.append(y)

    y_true = torch.cat(labels_list).detach().cpu().numpy()
    y_pred = torch.cat(preds_list).detach().cpu().numpy()
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')

    # Write into tensorboard
    # TODO: recore false-pos and false-neg samples.
    results = {
        'loss': eval_loss / eval_steps,
        'f1': f1_score,
        'precision': precision,
        'recall': recall
    }
    msg = f'*** Eval: loss {loss}, f1 {f1_score}, precision {precision}, recall {recall}'
    print(msg)
    return results




