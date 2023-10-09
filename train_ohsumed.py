# encoding=utf8
import os
from models.bert_classification import RoBertaClassifyAug
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
import argparse
import re
from pathlib import Path
from utils import sequence_padding, text_segmentate, clean_str
import time
from config import ohsumed_config as config
import random
from collections import Counter


torch.manual_seed(3407)  # pytorch random seed
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)

tokenizer = RobertaTokenizerFast.from_pretrained(config.pretrained_model_path)
maxlen = config.max_seq_len
categories = set()
labels_per_class = Counter()
if not os.path.exists(config.model_save_dir):
    os.makedirs(Path(config.model_save_dir).parent, exist_ok=True)


def load_20ng_data(filename, label_num_per_class=None, is_clean=None):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    if not label_num_per_class:
        sample_lens = []
        with open(filename, encoding='utf8') as f:
            for line in tqdm(f, "load data"):
                line = line.strip().split('|||')
                label, text = line
                if is_clean:
                    text = clean_str(text)
                sample_lens.append(len(text.split()))
                D.append({'text':text, 'label': label})
                categories.add(label)
        print(f"{filename}: maxlen {max(sample_lens)}, avglen {np.mean(sample_lens)}, minlen {min(sample_lens)}")
    else:
        with open(filename, encoding='utf8') as f:
            for line in f:
                line = line.strip().split('|||')
                label, text = line
                if labels_per_class[label] == label_num_per_class:
                    continue
                else:
                    D.append({'text': text, 'label': label})
                    categories.add(label)
                    labels_per_class[label] += 1
        print(labels_per_class)
    return D


train_data = load_20ng_data(config.train_data_path)
np.random.shuffle(train_data)
valid_data = load_20ng_data(config.dev_data_path)
np.random.shuffle(valid_data)
test_data = load_20ng_data(config.test_data_path)
np.random.shuffle(test_data)

categories = list(sorted(categories))
print(f'categories {categories} length {len(categories)}')


def mask_text(input_ids, mask_probability=0.15):
    text = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(len(text)):
        rand = random.random()
        if rand < mask_probability:
            if i == 0 or ('Ġ' in text[i]):
                text[i] = tokenizer.mask_token
                j = i+1
                while j < len(text) and 'Ġ' not in text[j]:
                    text[j] = tokenizer.mask_token
                    j += 1
    input_ids = tokenizer.convert_tokens_to_ids(text)

    return input_ids


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode='train'):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode

    def __getitem__(self, index):
        d = self.data[index]
        outputs = tokenizer(d['text'], return_offsets_mapping=True, truncation=True,
                            max_length=self.seq_len)
        token_ids = outputs['input_ids']
        if config.rank:
            token_ids_aug1 = mask_text(token_ids, 0.1)
        else:
            token_ids_aug1 = mask_text(token_ids)
        token_ids_aug2 = mask_text(token_ids)
        token_ids_aug3 = mask_text(token_ids)
        mask = outputs['attention_mask']
        label = F.one_hot(torch.tensor(categories.index(d['label'])), len(categories))

        return d, token_ids, token_ids_aug1, token_ids_aug2, token_ids_aug3, mask, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    d, token_ids, token_ids_aug1, token_ids_aug2, token_ids_aug3, mask, labels = list(zip(*batch))

    token_ids = torch.LongTensor(sequence_padding(token_ids))
    token_ids_aug1 = torch.LongTensor(sequence_padding(token_ids_aug1))
    token_ids_aug2 = torch.LongTensor(sequence_padding(token_ids_aug2))
    token_ids_aug3 = torch.LongTensor(sequence_padding(token_ids_aug3))
    labels = torch.FloatTensor(sequence_padding(labels))
    mask = torch.LongTensor(sequence_padding(mask))

    return d, token_ids, token_ids_aug1, token_ids_aug2, token_ids_aug3, mask, labels

train_dataset = TextSamplerDataset(train_data, maxlen)
valid_dataset = TextSamplerDataset(valid_data, maxlen)
test_dataset = TextSamplerDataset(test_data, maxlen)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)


def build_optimizer_and_scheduler(model, config, total_steps):

    module = (model.module if hasattr(model, "module") else model)
    model_param = list(module.named_parameters())

    bert_param = []
    other_param = []
    for name, param in model_param:
        if name.split('.')[0] == 'roberta':
            bert_param.append((name, param))
        else:
            other_param.append((name, param))

    no_decay = ["bias", "LayerNorm.weight", 'layer_norm']
    optimizer_grouped_parameters = [
        # bert module
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.lr},

        # other module
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.other_lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.other_lr}
    ]

    warmup_steps = int(config.warmup_proportion * total_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                          num_training_steps=total_steps, lr_end=config.lr_end)
    return optimizer, scheduler


def calculate_acc(y_pred, y_true):
    y_pred = torch.argmax(F.softmax(y_pred, dim=-1), dim=-1)
    y_true = torch.argmax(y_true, dim=-1)
    length = len(y_true)

    true_num = torch.sum(y_pred == y_true)
    return true_num, length

def train():
    print('config.aug', config.aug)
    model = RoBertaClassifyAug.from_pretrained(config.pretrained_model_path, label_num=len(categories),
                                               lamb=config.lamb)
    model.cuda()
    total_steps = len(train_loader) * config.max_epoches // config.gradient_accumulation_steps
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    if config.continue_train:
        model.load_state_dict(torch.load(config.model_save_dir))
    best_acc = 0.

    no_improvement_epochs = 0
    improve_flag = False
    for _ in range(config.max_epoches):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_true, total_length = 0, 0
        total_loss = 0
        for train_batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            x, x1, x2, x3, mask, labels = batch
            if config.aug:
                loss, logits = model(x, x1, x2, x3, mask=mask, target=labels)
            else:
                loss, logits = model(x, mask=mask, target=labels)
            true_num, current_length = calculate_acc(logits, labels)
            total_true += true_num
            total_length += current_length

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (train_batch_ind + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()  # update parameters of net
                scheduler.step()  # update learning rate schedule
                optimizer.zero_grad()  # reset gradients

            total_loss += loss.item()
            avg_loss = total_loss / (train_batch_ind + 1)
            avg_acc = total_true / total_length
            pbar.set_description(f"{_ + 1}/{config.max_epoches} Epochs")
            pbar.set_postfix(loss=avg_loss, acc=avg_acc.item(), lr=optimizer.param_groups[0]['lr'])

            if _ >= config.valid_start_epoch:
                if (train_batch_ind % np.ceil(len(train_loader) * config.valid_portion) == 0 and train_batch_ind != 0) or train_batch_ind == len(train_loader)-1:
                    avg_acc = evaluate(model, test_loader)

                    if avg_acc > best_acc:
                        if config.save_model:
                            if config.aug:
                                model_save_dir = config.model_save_dir.split('.')
                                torch.save(model.state_dict(), model_save_dir[0]+'_aug.'+model_save_dir[1])
                            else:
                                torch.save(model.state_dict(), config.model_save_dir)
                        print(f'best model saved with acc {avg_acc.item()}')
                        best_acc = avg_acc
                        no_improvement_epochs = 0
                        improve_flag = True

                    model.train()
        if improve_flag:
            no_improvement_epochs = 0
            improve_flag = False
        else:
            no_improvement_epochs += 1
        print(f'{no_improvement_epochs} epochs have no improvements, reach {config.early_stop} will stop training')
        if no_improvement_epochs == config.early_stop:
            break

def evaluate(model, data_loader):
    model.eval()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    total_true, total_length = 0, 0
    with torch.no_grad():
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            x, x1, x2, x3, mask, labels = batch
            loss, logits = model(x, mask=mask, target=labels)
            true_num, current_length = calculate_acc(logits, labels)
            total_true += true_num
            total_length += current_length
            avg_acc = total_true / total_length
            pbar.set_postfix(acc=avg_acc.item())
    return avg_acc


def test():
    model = RoBertaClassifyAug.from_pretrained(config.pretrained_model_path, label_num=len(categories))
    model.cuda()

    if config.aug:
        model_save_dir = config.model_save_dir.split('.')
        model.load_state_dict(torch.load(model_save_dir[0] + '_aug.' + model_save_dir[1]))
    else:
        model.load_state_dict(torch.load(config.model_save_dir))
    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    total_true, total_length = 0, 0
    with torch.no_grad():
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            x, x1, x2, x3, mask, labels = batch
            loss, logits = model(x, mask=mask, target=labels)
            true_num, current_length = calculate_acc(logits, labels)
            total_true += true_num
            total_length += current_length
            avg_acc = total_true / total_length
            pbar.set_postfix(acc=avg_acc.item())
    print(f'test acc {avg_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gau')
    parser.add_argument('-m',
                        type=str,
                        default='test',
                        help='train or eval')
    parser.add_argument('-r',
                        type=str,
                        default='all',
                        help='part or all')
    parser.add_argument('--eval_num',
                        type=int,
                        default=10,
                        help='eval number')
    parser.add_argument('--device_id',
                        type=int,
                        default=3,
                        help='device id')
    parser.add_argument('--aug',
                        action="store_true",
                        help='is augment')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    config.aug = args.aug
    print(config.__dict__)
    if args.m == 'train':
        train()
    else:
        test()