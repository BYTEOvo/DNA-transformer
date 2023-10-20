# coding: utf-8
import argparse
import time
import datetime
import math
import os
import itertools
import json
import numpy as np
import pandas as pd

from utils.dna_functions import Logger, FocalLoss

import torch
import torch.optim as optim
import torch.nn as nn

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data_path', type=str,
                    help='dataset path')
parser.add_argument('--m_type', type=str,
                    help='model type')
parser.add_argument('--output_path', type=str, default='output/',
                    help='output path')
parser.add_argument('--emb_path', type=str, default=None,
                    help='embedding path')
parser.add_argument('--n_layer', type=int, default=6,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=4,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=32,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=64,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=256,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--optim_type', type=int, default=0,
                    help='define optimizer type')
parser.add_argument('--shift', type=int, default=20,
                    help='shift labels')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--conv_size', type=int, default=7,
                    help='motif size of the conv layer')
parser.add_argument('--conv_emb', action='store_true',
                    help='convolutional embedding')
parser.add_argument('--pre_conv', action='store_true',
                    help='convolution before QKV')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--ext_ds', type=int, default=0,
                    help='number of tokens to include downstream of label')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--methylation', action='store_true',
                    help='use methylation patterns -> 8 input classes')
args = parser.parse_known_args()[0]
# print(args)
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0


args.work_dir = os.path.join('{}-{}'.format(args.output_path, args.data_path.split('/')[-1]),
                             time.strftime('%Y%m%d-%H%M%S'))

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
n_token_out = 2
n_token_in = 4 + 4 * args.methylation
eval_batch_size = args.batch_size

corpus = get_lm_corpus(args.data_path, labels=n_token_out, merge_size=1, shift=args.shift)
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
args.coords = [corpus.Gen_train.start, corpus.Gen_train.end, corpus.Gen_test.start, corpus.Gen_test.end,
               corpus.Gen_valid.start, corpus.Gen_valid.end]
# print(f'Train set: {args.coords[0]}-{args.coords[1]}\t Test set: {args.coords[2]}-{args.coords[3]}\t Valid set: '
#       f'{args.coords[4]}-{args.coords[5]}')


def evaluate(model, eval_iter, args, criterion=None, weights_optim=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.same_length = True
    model.reset_length(args.eval_tgt_len, args.eval_tgt_len, args.ext_ds)

    # Evaluation
    total_len, total_loss = 0, 0.
    probs, targets = [], []
    i_max = eval_iter.n_batch_same_length

    with torch.no_grad():
        mems = tuple()
        for i, (data, target_init, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            # Ensures all data is retained to reconstruct original genomic strand
            last = True if (i + 1 == i_max) else False
            ret = model(data, target_init, *mems, criterion=criterion, last=last)

            loss, prob, target, mems = ret[0], ret[1], ret[2], ret[3:]

            # transforming of data necessary to reconstruct original genomic sequence
            probs.append([p.reshape(-1, args.batch_size,
                                    p.shape[1]).transpose(1, 0, 2) for p in prob])
            targets.append([t.reshape(-1, args.batch_size).T for t in target])

            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # transforming of data necessary to reconstruct original genomic sequence
    probs = [np.concatenate([p[i] for p in probs], axis=1) for i in range(len(probs[0]))]
    targets = [np.concatenate([t[i] for t in targets], axis=1) for i in range(len(targets[0]))]

    probs = [p.reshape(-1, p.shape[2]) for p in probs]
    targets = [t.reshape(-1) for t in targets]

    model.same_length = False
    model.reset_length(args.tgt_len, args.mem_len, args.ext_ds)
    model.train()

    return total_loss / total_len, probs, targets



args.criterion_gamma = 0.8 # not a big effect
criterion = FocalLoss(gamma=args.criterion_gamma)

model_name = ""
if args.m_type == "TIS":
    model_name = "TIS"
    print("TIS")
elif args.m_type == "M4C":
    model_name = "M4C"
    print("M4C")
elif args.m_type == "TSS":
    model_name = "TSS"
    print("TSS")

model_name = model_name + '_model.pt'
with open(os.path.join('./', 'models/', model_name), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)
test_loss, test_probs, test_targets = evaluate(para_model, te_iter, args, criterion=criterion)
print("test_targets~~~~~~~~~")
indices = []
for i in range(len(test_targets[0])):
    if test_targets[0][i] == 1:
        indices.append(i)

print("@",indices,"@")
# print(len(indices))
# 打开txt文件作为写入模式
with open('output.txt', 'w') as file:
    # 将整数数组转换为字符串数组
    str_targets = [str(target) for target in test_targets[0]]
    # 将字符串数组连接成一个字符串，每个元素用换行符分隔
    content = '\n'.join(str_targets)
    # 将字符串写入文件
    file.write(content)