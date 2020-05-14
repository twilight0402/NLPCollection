import os
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
# from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange       # 进度条
from sklearn.metrics import precision_recall_fscore_support
from classification.models.LSTM_model import LSTM_model
from classification.utils.tool import build_and_cache_dataset, save_model
from classification.configure import Config


def train(config: Config):

    # Build train dataset 获得数据集
    fields, train_dataset = build_and_cache_dataset(config, mode='train')

    # Build vocab 获得词向量
    ID, CATEGORY, NEWS = fields
    # 词向量是文本文件，共365113个字，第一行为介绍，第一列是词，后面的列为向量，空格分割
    vectors = Vectors(name=config.embedding_dir, cache=args.data_dir)
    # NOTE: use train_dataset to build vocab!
    NEWS.build_vocab(
        train_dataset,
        max_size=config.vocab_size,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )
    CATEGORY.build_vocab(train_dataset)

    model = LSTM_model(
        vocab_size=len(NEWS.vocab),     # 词表的长度， 8188
        output_dim=config.num_labels,     # 输出的维度， 默认5
        pad_idx=NEWS.vocab.stoi[NEWS.pad_token],    # 填充的id
        dropout=config.dropout,
    )
    # Init embeddings for model
    model.embedding.from_pretrained(NEWS.vocab.vectors)

    bucket_iterator = BucketIterator(
        train_dataset,
        batch_size=args.train_batch_size,
        sort_within_batch=True,
        shuffle=True,
        sort_key=lambda x: len(x.news),     # 排序是后面的pack需要用到
        device=args.device,
    )

    # optimizer, lr_scheduler, criterion
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     eps=args.adam_epsilon)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.learning_rate * 10,
                           epochs=args.num_train_epochs,
                           steps_per_epoch=len(bucket_iterator))

    global_step = 0
    model.zero_grad()
    train_trange = trange(0, args.num_train_epochs, desc="Train epoch")
    for _ in train_trange:
        epoch_iterator = tqdm(bucket_iterator, desc='Training')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            news, news_lengths = batch.news
            category = batch.category
            preds = model(news, news_lengths)

            loss = criterion(preds, category)
            loss.backward()

            # Logging
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/lr',
                              scheduler.get_last_lr()[0], global_step)

            # NOTE: Update model, optimizer should update before scheduler
            optimizer.step()
            scheduler.step()
            global_step += 1

            # NOTE:Evaluate
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = evaluate(args, model, CATEGORY.vocab, NEWS.vocab)
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value,
                                      global_step)

            # NOTE: save model
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, optimizer, scheduler, global_step)

    writer.close()


def evaluate(args, model, category_vocab, example_vocab, mode='dev'):
    fields, eval_dataset = build_and_cache_dataset(args, mode=mode)
    # 没有shuffle=True了
    bucket_iterator = BucketIterator(
        eval_dataset,
        train=False,
        batch_size=args.eval_batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.news),
        device=args.device,
    )
    ID, CATEGORY, NEWS = fields
    CATEGORY.vocab = category_vocab
    NEWS.vocab = example_vocab
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # NOTE: Eval!
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss, eval_steps = 0.0, 0
    labels_list, preds_list = [], []
    for batch in tqdm(bucket_iterator, desc='Evaluation'):
        news, news_lengths = batch.news
        labels = batch.category
        with torch.no_grad():
            logits = model(news, news_lengths)
            loss = criterion(logits, labels)
            eval_loss += loss.item()

        eval_steps += 1
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds)
        labels_list.append(labels)

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
    logger.info(msg)
    return results


def main():
    args = get_args()
    writer = SummaryWriter()

    # Check output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError(f"output目录({args.output_dir}) 已经存在并且不为空 "
                         "使用 --overwrite_output_dir")

    # 选择设备， 可在参数中指定是否使用cuda (no_cuda)
    device = "cuda" if torch.cuda.is_available() \
            and not args.no_cuda else "cpu"
    args.device = torch.device(device)      # 把device带入到args中，用来传递
    logger.info("使用设备: %s", device)

    # 开始训练
    train(args, writer)


if __name__ == "__main__":
    main()


