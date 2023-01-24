#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:49:47 2022

@author: sho
"""

import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
from OmegaPLM import OmegaPLM
import argparse
import math
from einops import rearrange
from functools import reduce
import Bio
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
import warnings
warnings.simplefilter('ignore')

nu_list = {"input": ["A", "C", "G", "T"], "output": ["A", "C", "G", "T", "X"]}

def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
        data.append([record.id, record.seq])
    return pd.DataFrame(data, columns = ["id", "seq"])


def numbering(seq, mode = "input"):
    return np.array([nu_list[mode].index(seq[i]) if seq[i] in nu_list[mode] else len(nu_list[mode]) for i in range(len(seq))])

class bsampler(data.Dataset):
    def __init__(self, data_sets, max_seq_len, device = "cuda:0"):
        super().__init__()
        self.max_len = max_seq_len
        
        self.seq = [data_sets.loc[i, "seq"]  for i in range(len(data_sets)) if len(data_sets.loc[i, "seq"]) < max_seq_len]
        
        self.device = device

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        node_input_fe = numbering(self.seq[idx], mode = "input")
        
        mask = torch.concat([torch.ones([len(node_input_fe), 1]), torch.zeros([self.max_len - len(node_input_fe), 1])], axis = 0)
        
        node_input_fe = node_input_fe.reshape([len(node_input_fe), 1])
        node_input_fe = np.concatenate([node_input_fe, np.full((self.max_len - len(node_input_fe), 1), len(nu_list["input"]) + 1)], axis = 0)
        
        return torch.tensor(node_input_fe).long().to(self.device).squeeze(), torch.tensor(mask).long().to(self.device).squeeze()

def _make_config(input_dict: dict) -> argparse.Namespace:
    """Recursively go through dictionary"""
    new_dict = {}
    for k, v in input_dict.items():
        if type(v) == dict:
            new_dict[k] = _make_config(v)
        else:
            new_dict[k] = v
    return argparse.Namespace(**new_dict)

def mlm(x, word_pred = 0.15, pred_probs = [0.8, 0.1, 0.1], pad_index = 0, mask_index = 0, n_words = 4, device = "cpu"):
    #x = x.T
    batch, seq_length = x.size()

    pred_mask = np.random.rand(batch, seq_length) <= word_pred
    pred_mask = torch.from_numpy(pred_mask.astype(np.uint8)).to(device)
    
    pred_mask[x == pad_index] = 0
    x_real = x[pred_mask]
    
    x_rand = x_real.clone().random_(n_words)
    x_mask = x_real.clone().fill_(mask_index)
    
    probs = torch.multinomial(pred_probs, len(x_real), replacement=True).to(device)
    x_replace = x_mask * (probs == 0).long() + x_real * (probs == 1).long() + x_rand * (probs == 2).long()
    x = x.masked_scatter(pred_mask, x_replace)
    
    return x.to(device), x_real.to(device), pred_mask.to(device)

class MLMProcess():
    def __init__(self, out_path, train_params, device):
        self.out_path = out_path
        self.train_params = train_params
        self.device = device
        
    def model_training(self, data_sets):
        print("data size:: {}".format(len(data_sets)), flush = True)
        
        os.makedirs(self.out_path, exist_ok=True)
        sampler = bsampler(data_sets, self.train_params.max_seq, device = device)
        loader = DataLoader(dataset = sampler, batch_size = self.train_params.batch)

        cfg = dict(
               plm=dict(
                   alphabet_size=len(nu_list["input"]) + 3,
                   #node=1280,
                   node=320,
                   padding_idx=len(nu_list["input"]) + 1,
                   out_size = len(nu_list["output"]),
                   edge=10,
                   #proj_dim=1280 * 2,
                   proj_dim=320 * 2,
                   #attn_dim=256,
                   attn_dim=64,
                   num_head=1,
                   num_relpos=129,
                   masked_ratio=0.12,
               )
            ) 
        
        word_pred = self.train_params.word_pred
        pred_probs = self.train_params.pred_probs
        pad_index = self.train_params.pad_index
        mask_index = self.train_params.mask_index
        n_words = self.train_params.n_words
       
        cfg = _make_config(cfg)
        self.fwd_cfg = _make_config(dict(subbatch_size = None))
        self.model = OmegaPLM(cfg.plm).to(self.device)
        self.opt = optim.Adam(params = self.model.parameters(), lr = self.train_params.lr)
        criteria = nn.BCELoss()
        
        for epoch in range(self.train_params.max_epoch):
            loss_list = []
            print("Epoch_{}=============================".format(epoch + 1), flush = True)
            
            self.model.train()
            for i, (node_input_fe, mask) in enumerate(loader):
                self.opt.zero_grad()
                masked_seq, label, masked_index = mlm(node_input_fe, word_pred = word_pred, pred_probs = pred_probs, pad_index = pad_index, mask_index = mask_index, n_words = n_words, device = self.device)
                
                pred_seq = self.model(masked_seq, mask, fwd_cfg = self.fwd_cfg)
                
                pred = pred_seq[masked_index]
                label = label.reshape([len(label), 1])
                pred = pred.gather(1, label)
                
                loss = criteria(pred, torch.ones_like(pred))
               
                loss.backward()
                self.opt.step()
                loss_list.append(loss.item())
            
            loss_list = np.mean(loss_list)
            print("loss:: " + str(loss_list), flush = True)
            
            if((epoch + 1) % 10 == 0):
                os.makedirs(self.out_path + "/data_model", exist_ok=True)
                torch.save(self.model.state_dict(), "{}/data_model/deep_model_epoch_{}.pt".format(self.out_path, epoch + 1))
                

in_path = "/home/dl-box/Research/RNA_secondary_structure_pred/ver22/data/non_red_ncrna_NONCODEv5.0_extract.fasta"
out_path = "/home/dl-box/Research/RNA_secondary_structure_pred/ver22/data/res/mlm"

device = "cpu"

#word_pred = 0.15 #maskするtokenの割合
word_mask = 0.8 #maskするtokenのうちmaskに置き換えるtokenの割合
word_keep = 0.1 #maskするtokenのうち置き換えをおこわないtokenの割合
word_rand = 0.1 #maskするtokenのうちrandomに置き換えるtokenの割合

train_params = dict(
    lr=0.00001,
    batch=64,
    val_batch=64,
    max_epoch=20000,
    max_seq=81,
    word_pred=0.15, #maskするtokenの割合
    pred_probs=torch.FloatTensor([word_mask, word_keep, word_rand]), #maskトークンのうちmaskに置き換えるtokenの割合、置き換えをおこわないtokenの割合、randomに置き換えるtokenの割合を設定
    pad_index=len(nu_list["input"]) + 1, #長さ調節のためのpaddingのid
    mask_index=len(nu_list["input"]) + 2, #masked language modelingのmaskのid
    n_words=len(nu_list["input"]) #tokenの種類数 (A, U, G, C)
    )
train_params = _make_config(train_params)

data = open_fasta(in_path)

mlm_obj = MLMProcess(out_path, train_params, device)
temp = mlm_obj.model_training(data)









