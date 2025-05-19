import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torch.nn import DataParallel
from transformers import AutoModel
from pyclustering.cluster.kmeans import kmeans, distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from metric import SpanEntityScore, print_table
from tqdm import tqdm
from common import seed_everything
from copy import deepcopy

import faiss
import numpy as np
import json
import math
import random
import os
import time
import dill as pickle


class MyIter:
    def __init__(self, iters):
        self.iters = iters

    def __iter__(self):
        for i in tqdm(self.iters, ncols=50):
            yield i


class NerDataLoader(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        label_dir,
        max_length=100,
        data_size=1,
        model_type=None,
        markup='bio'
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.max_length = max_length
        self.data_size = data_size
        self.model_type = model_type
        self.markup = markup
        self.is_cls = '[CLS]' in tokenizer.get_vocab()
        self.texts = None
        self.pinyin_ids = None
        self.labels_locate = None
        self.labels_type = None
        self.masks = None
        self.label2idx = None
        self.idx2label = None
        self.max_len = 0

    def get_soups(self, dev=False):
        self.get_labels()
        self.load()
        self.tokenize(dev)
        return self.label2idx, self.idx2label

    def get_labels(self):
        with open(self.label_dir, 'r', encoding='utf8') as f:
            idx2label = f.read().strip().split('\n')
        if self.markup == 'io':
            idx2label = [i for i in idx2label if not i.startswith('B-')]
        self.idx2label = idx2label
        self.label2idx = {k: v for v, k in enumerate(idx2label)}

    def load(self):
        print("Loading dataset...")
        texts, labels = [], []
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            all = f.read().strip().split('\n')
        all.append('')
        text, label = [], []
        for d in tqdm(all, ncols=50):
            if d == '':
                if random.uniform(0, 1) > self.data_size:
                    continue
                texts.append(text)
                labels.append(label)
                text, label = [], []
            else:
                text.append(d.split('\t')[0])  # uncased使用
                if self.markup == 'io':
                    th = d.split('\t')[-1]
                    label.append(
                        self.label2idx[th if not th.startswith('B-') else ('I' + th[1:])])
                else:
                    label.append(self.label2idx[d.split('\t')[-1]])
        self.texts = texts
        self.pinyin_ids = [torch.tensor([])] * len(texts)
        self.labels_type = labels
        self.labels_locate = [[] for _ in range(len(labels))]
        self.masks = [[] for _ in range(len(labels))]

    def tokenize(self, dev=False):
        print("Tokenizing...")
        for idx, (text, label) in enumerate(zip(self.texts, self.labels_type)):
            tokens, labels = [], []
            for t, l in zip(text, label):
                token = self.tokenizer.encode(t, add_special_tokens=False)
                tokens.extend(token)
                labels.extend([l] * len(token))

            self.max_len = max(self.max_len, len(tokens))
            if dev:
                if self.is_cls:
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"]) + tokens[:510] + self.tokenizer.convert_tokens_to_ids(["[SEP]"])
                else:
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        ["<s>"]) + tokens[:510] + self.tokenizer.convert_tokens_to_ids(["</s>"])
                self.masks[idx] = torch.tensor([1] * len(tokens))
                labels_type = [0] + labels[:510] + [0]
            else:
                self.masks[idx] = torch.tensor(
                    [1] * (len(tokens[:self.max_length]) + 2) + [0] * (self.max_length - len(tokens)))
                labels_type = [0] + labels[:self.max_length] + \
                    [0] + [0] * (self.max_length - len(tokens))
                if self.is_cls:
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"]) + tokens[:self.max_length] + self.tokenizer.convert_tokens_to_ids(["[SEP]"]) + self.tokenizer.convert_tokens_to_ids(["[PAD]"]) * (self.max_length - len(tokens))
                else:
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        ["<s>"]) + tokens[:self.max_length] + self.tokenizer.convert_tokens_to_ids(["</s>"]) + self.tokenizer.convert_tokens_to_ids(["<pad>"]) * (self.max_length - len(tokens))

            labels_locate, lens = [0] * len(tokens), len(labels_type)
            for id, l in enumerate(labels_type):
                if self.idx2label[l].startswith('B-'):
                    if id == lens-1 or self.idx2label[labels_type[id+1]].startswith('B-'):
                        labels_locate[id] = 3
                        continue
                    labels_locate[id] = 1
                elif self.idx2label[l].startswith('I-'):
                    if id == lens-1 or self.idx2label[labels_type[id+1]].startswith('B-') or self.idx2label[labels_type[id+1]].startswith('O'):
                        labels_locate[id] = 2
            self.texts[idx] = torch.tensor(tokens)
            if self.model_type in ['chinesebert-base', 'chinesebert-large']:
                self.pinyin_ids[idx] = torch.tensor(
                    self.tokenizer.convert_ids_to_pinyin_ids(tokens))
            self.labels_type[idx] = torch.tensor(labels_type)
            self.labels_locate[idx] = torch.tensor(labels_locate)

    def __getitem__(self, index):
        return self.texts[index], self.pinyin_ids[index], self.labels_type[index], self.labels_locate[index], self.masks[index], index

    def __len__(self):
        return len(self.texts)


class NerModel(nn.Module):

    def __init__(
        self,
        vec_dir,
        num_labels,
        dropout_locate=0.1,
        dropout_type=0.2,
        model_type=None
    ):
        super(NerModel, self).__init__()
        self.num_labels = num_labels
        if model_type == 'chinesebert-base':
            self.bert = AutoModel.from_pretrained(
                'iioSnail/ChineseBERT-base', trust_remote_code=True, cache_dir=vec_dir, add_pooling_layer=False)
        elif model_type == 'chinesebert-large':
            self.bert = AutoModel.from_pretrained(
                'iioSnail/ChineseBERT-large', trust_remote_code=True, cache_dir=vec_dir, add_pooling_layer=False)
        else:
            self.bert = AutoModel.from_pretrained(
                vec_dir, add_pooling_layer=False)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.locate_dropout = nn.Dropout(dropout_locate)
        self.type_dropout = nn.Dropout(dropout_type)
        self.model_type = model_type
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.locate_mlp = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.locate_cls = nn.Linear(self.bert.config.hidden_size, 4)
        self.type_cls = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input, label=None):
        text, pinyin, attention_mask = input
        if self.model_type in ['chinesebert-base', 'chinesebert-large']:
            hiddens = self.bert(
                text, attention_mask=attention_mask, pinyin_ids=pinyin)[0]
        else:
            hiddens = self.bert(
                text, attention_mask=attention_mask)[0]
        logits = self.type_cls(self.type_dropout(hiddens))
        locate_logits = self.locate_cls(
            self.norm(self.locate_dropout(torch.tanh(self.locate_mlp(hiddens))) + hiddens))
        # locate_logits = self.locate_cls(self.locate_dropout(hiddens))

        return hiddens, logits, locate_logits

    def update(self, num_labels):
        self.num_labels = num_labels
        self.type_cls = nn.Linear(
            self.bert.config.hidden_size, num_labels).cuda()

    def save(self, model, save_dir, model_type, max_k, coarse=''):
        model_name = "{}_{}_{}.pkl".format(
            model_type, max_k, coarse
        )
        with open(os.path.join(save_dir, model_name), "wb") as f:
            pickle.dump(model, f)
        return os.path.join(save_dir, model_name)


class KMean:
    def __init__(self, dataset, dataloader, max_length, max_k=20, minibatch_size=2024, λ=0.5, t=1):
        self.initializer = None
        self.dataset = dataset
        self.dataloader = dataloader
        self.max_length = max_length
        self.max_k = max_k
        self.minibatch_size = minibatch_size
        self.vecs = None
        self.label2sub = {}
        self.sub2label = {}
        self.num_labels = 0
        self.sub_labels = 0
        self.λ = λ
        self.t = t

    def get_vecs(self, model):
        vecs = {i: [] for i in range(model.module.num_labels if isinstance(
            model, DataParallel) else model.num_labels)}
        with torch.inference_mode(True):
            for idx_d, (text, pinyin, label, _, mask, idx) in enumerate(self.dataloader):
                # text, pinyin, label, mask = text.cuda(), pinyin.cuda(), label.cuda(), mask.cuda()
                text, pinyin, label, mask = text.cuda(), pinyin.cuda(), label.cuda(), mask.cuda()
                with autocast():
                    # hidden : [batch_size, seq_len, 768], locate_label : [batch_size, seq_len, 4]
                    hidden, _, _ = model((text, pinyin, mask), label)
                hidden = F.normalize(hidden, dim=-1)

                type_data, type_label = hidden[mask == 1].tolist(
                ), label[mask == 1].tolist()
                assert len(type_data) == len(type_label)
                for d, l in zip(type_data, type_label):
                    t = int(l)
                    vecs[t].append(d)
        self.vecs = vecs

    def update(self, model):
        '''
        通过model更新聚类点，聚类点是label与点对应
        '''
        print("Update initial points...")
        start = time.time()
        self.get_vecs(model=model)

        def cos_metric(a, b):
            '''聚类：越近越小最好'''
            return 1 - self.cosine_similarity(a, b)
        initial_centers, dis_metric = {}, distance_metric(
            metric_type=type_metric.USER_DEFINED, func=cos_metric)

        self.label2sub = {}
        self.sub2label = {}
        self.num_labels = 0
        self.sub_labels = 0
        iter_max = 100
        epsilon = 0.9
        for key, val in self.vecs.items():
            print("coarse_type id: {} point_count: {}".format(key, len(val)))
            if self.initializer != None:
                km = kmeans(
                    val, self.initializer[key], metric=dis_metric, itermax=iter_max)
                km.process()
                initial_centers[key] = km.get_centers()
                initial_centers[key] = self.merge(
                    initial_centers[key], epsilon)
                k_ = len(initial_centers[key])
                self.num_labels += 1
                self.sub_labels += k_

                self.label2sub[key] = list(
                    range(self.label2sub[key-1][-1] + 1, self.label2sub[key-1][-1] + k_ + 1)) if key else list(range(0, k_))
                for l in self.label2sub[key]:
                    self.sub2label[l] = key
                print('pre_k_:', len(self.initializer[key]), 'now_k_:', len(
                    initial_centers[key]))
            else:
                k_, val_ = 1, deepcopy(val)
                if key:
                    k_ = optimal_k(val, max_k=min(self.max_k, len(
                        val_)-1), batch_size=self.minibatch_size, iter_max=100)
                    k_ = 1 if k_ <= 2 else k_

                self.num_labels += 1
                self.sub_labels += k_

                self.label2sub[key] = list(
                    range(self.label2sub[key-1][-1] + 1, self.label2sub[key-1][-1] + k_ + 1)) if key else list(range(0, k_))
                for l in self.label2sub[key]:
                    self.sub2label[l] = key
                initial = kmeans_plusplus_initializer(
                    np.array(val), k_).initialize()
                km = kmeans(val, initial, metric=dis_metric, itermax=iter_max)
                km.process()
                initial_centers[key] = km.get_centers()
                initial_centers[key] = self.merge(
                    initial_centers[key], epsilon)
                print("pre_type k_: ", k_, 'now_k_:',
                      len(initial_centers[key]))
        self.initializer = initial_centers
        print("label2sub", self.label2sub, "sub_labels_nums", self.sub_labels)
        print('Update kmean centers success with {:.4f} s'.format(
            time.time() - start))

    def merge(self, vecs, epsilon=0.9):
        '''
        合并大于阈值的表征向量
        '''
        res = []
        vis = [False for _ in range(len(vecs))]
        for i in range(len(vecs)):
            if vis[i]:
                continue
            for j in range(i+1, len(vecs)):
                if not vis[j] and self.cosine_similarity(vecs[i], vecs[j]) > epsilon:
                    vis[j] = True
                    vis[i] = True
                    res.append([(vecs[i][idx] + vecs[j][idx]) /
                               2 for idx in range(len(vecs[i]))])
                    break
            if not vis[i]:
                res.append(vecs[i])
        return deepcopy(res)

    def get_sub(self, vec, label: int):
        vecs = np.array(self.initializer[label])
        vec = F.normalize(vec, dim=-1).cpu().detach().numpy()

        similarities = np.array([self.cosine_similarity(vecs[i], vec)
                                for i in range(vecs.shape[0])])
        idx = np.argmax(similarities)
        return self.label2sub[label][idx]

    def cosine_similarity(self, a, b):
        '''
        先归一化再求余弦相似度，和如下求法一致
        '''
        a = np.array(a)
        b = np.array(b)
        dot_product = np.dot(a, b.T)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)


def optimal_k(data, max_k, batch_size=500, iter_max=100):
    iters = range(2, max_k+1)
    if max_k <= 2:
        return max_k
    s = []
    for k in tqdm(iters, ncols=50, desc='Selecting K'):
        # scikit-learn >= 1.3.2 when n_init='auto'
        # kmeans = KMeans(n_clusters=k, init='k-means++',
        #                 max_iter=200, n_init="auto")
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', random_state=42,
                                 max_iter=iter_max, n_init='auto', batch_size=batch_size)
        kmeans.fit(data)
        # 支持评价标准：[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
        s.append(silhouette_score(data, kmeans.labels_, metric='cosine'))

    optimal_k = iters[s.index(max(s))]

    return optimal_k


class Evaluater:
    def __init__(self, dev_dataset, is_nest, model_dir, locate_td=0, type_td=0, markup='bio'):
        self.dev_dataset = dev_dataset
        self.is_nest = is_nest
        self.model_dir = model_dir
        self.model = None
        self.locate_td = locate_td
        self.type_td = type_td
        self.markup = markup

    def load_model(self):
        print("Loading evaluate model...")
        start = time.time()
        self.model = pickle.load(open(self.model_dir, "rb"))
        end = time.time()
        print("Loading evaluate model time: {:.4f} s".format(end - start))
        return self.model

    def evaluate_ner(self, km: KMean = None):
        print('Evaluating...')
        dataloader = DataLoader(self.dev_dataset, shuffle=False)
        pre_labels, real_labels = [], []
        locate_pre_labels, locate_real_labels = [], []
        self.model.eval()
        with torch.inference_mode(mode=True):
            for idx_d, (text, pinyin, label, label_loc, mask, idx) in enumerate(dataloader):

                text, pinyin, label, label_loc, mask = text.cuda(
                ), pinyin.cuda(), label.cuda(), label_loc.cuda(), mask.cuda()
                with autocast():
                    hidden, type_label, locate_label = self.model(
                        (text, pinyin, mask), label)

                locate_pre_labels.extend(
                    locate_label.argmax(dim=-1)[0].tolist())
                locate_real_labels.extend(label_loc[0].tolist())

                type_pre_logits = F.softmax(type_label, dim=-1)[0]
                # knn slot
                pre_label = torch.argmax(
                    type_pre_logits, dim=-1).tolist()  # [batch_size, 1]
                real_label = label[0].tolist()

                pre_labels.append(pre_label)
                real_labels.append(real_label)

        span = SpanEntityScore(
            id2label=self.dev_dataset.idx2label, is_nest=self.is_nest, markup=self.markup)
        span.update(real_labels, pre_labels)
        score, info = span.result()
        print_table(info)

        f1, precision, recall = score['f1'], score['precision'], score['recall']
        print('precision_score:{:.4f} recall_score:{:.4f} f1_score:{:.4f} '.
              format(precision, recall, f1))
        return f1
