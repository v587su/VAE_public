import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import json
import os
from torch.optim import Adam
from model import *
from dataset import QueryDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm


def train(args):
    with open('{}/word_vocab.json'.format(args.train_data_dir), 'r') as f:
        word_vocab = json.load(f)

    vocab_size = len(word_vocab)
    print('cuda:', args.with_cuda)
    cuda_condition = args.with_cuda == 1
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    corpus_train = QueryDataset('{}/query_corpus.txt'.format(args.train_data_dir),
                                word_vocab, args.query_len)

    corpus_val = QueryDataset('{}/query_corpus.txt'.format(args.train_data_dir),
                              word_vocab, args.query_len)

    train_data_loader = DataLoader(
        corpus_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    model = VAE(vocab_size, args.hidden_size, args.emb_size,dropout=args.dropout)
    print("Total Parameters:",
          sum([p.nelement() for p in model.parameters()]))

    loss = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr,
                 betas=[args.adam_beta1, args.adam_beta2],
                 weight_decay=args.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda
                                                  epoch: args.decay_ratio ** epoch)

    use_parallel = False
    if cuda_condition:
        if torch.cuda.device_count() > 1:
            use_parallel = True
            model = nn.DataParallel(model)
        model = model.to(device)

    kld_weight = args.init_kld_weight

    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()
        data_iter = tqdm.tqdm(enumerate(train_data_loader),
                              total=len(train_data_loader),
                              bar_format="{l_bar}{r_bar}")
        for i, (data, target) in data_iter:
            data = data.to(device)
            target = target.to(device)
            if use_parallel:
                m, l, z, decoded = model.module.forward(data)
            else:
                m, l, z, decoded = model.forward(data)

            loss_value = loss(decoded.view(-1, vocab_size), target.view(-1))
            KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) -
                                    torch.exp(l) + 1, 1)).mean().squeeze()
            loss_value += KLD * kld_weight
            if epoch > args.kld_start_inc and kld_weight < args.kld_max:
                kld_weight += args.kld_inc

            optim.zero_grad()
            loss_value.backward()
            if args.clip_grad_norm is not None:
                clip_grad_norm(model.parameters(), args.clip_grad_norm)
            optim.step()
            total_loss += float(loss_value.item())

            if i % args.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": total_loss / (i + 1),
                    "loss": loss_value.item(),
                    "kld": KLD.item()
                }
                data_iter.write(str(post_fix))

            if i % args.val_freq == 0:
                val_loss = validate(args, corpus_val, model,
                                    loss, use_parallel, vocab_size)
                post_fix = {
                    "val_loss": val_loss,
                }
                data_iter.write(str(post_fix))
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                with open('{}/performance.txt'.format(args.save_dir), 'a') as f:
                    if epoch == 0:
                        f.write(str(args)+'\n')
                    post_fix['epoch'] = epoch
                    f.write(str(post_fix) + '\n')
                f.close()
        scheduler.step()
        model_save_dir = '{}/main_model'.format(args.save_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        output_path = model_save_dir + "/main.model.ep%d" % epoch
        torch.save(model.cpu(), output_path)
        model.to(device)


def validate(args, valid_set, model, loss, use_parallel, vocab_size):
    model.eval()
    device = next(model.parameters()).device
    data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)
    total_loss = 0
    for data, target in tqdm.tqdm(data_loader):
        if args.with_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            if use_parallel:
                m, l, z, decoded = model.module.forward(data)
            else:
                m, l, z, decoded = model.forward(data)
            loss_value = loss(decoded.view(-1, vocab_size), target.view(-1))
            total_loss += loss_value.item()

    model.train()
    return total_loss / len(data_loader)


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_dir", type=str,
                        default="./data")
    parser.add_argument("--save_dir", type=str,
                        default="./output")

    parser.add_argument("-ql", "--query_len", type=int, default=20)
    parser.add_argument("-es", "--emb_size", type=int, default=128)
    parser.add_argument("-hs", "--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("-dp", "--dropout", type=float, default=0.1)

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-w", "--num_workers", type=int, default=0,)

    parser.add_argument('-cuda', "--with_cuda", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=100000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_ratio", type=float, default=0.95)
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--adam_weight_decay", type=float, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    parser.add_argument("--kld_start_inc", type=int, default=200)
    parser.add_argument("--init_kld_weight", type=float, default=0.05)
    parser.add_argument("--kld_max", type=float, default=0.1)
    parser.add_argument("--kld_inc", type=float, default=0.000002)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    print(args)
    train(args)
