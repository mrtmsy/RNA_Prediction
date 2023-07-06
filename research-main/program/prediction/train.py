import time
import os
import pandas as pd
import Bio
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from einops import rearrange
import math
from model_attn import Net
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}
import warnings
warnings.filterwarnings('ignore')

nu_list = {"input": ["A", "C", "G", "U"], "output": ["A", "C", "G", "U", "."]}

def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
        data.append([record.id, record.seq[0:81]])
    return pd.DataFrame(data, columns = ["id", "seq"])

def numbering(seq, mode = "input"):
    return np.array([nu_list[mode].index(seq[i]) if seq[i] in nu_list[mode] else len(nu_list[mode]) for i in range(len(seq))])

class bsampler(data.Dataset):
    def __init__(self, data_sets, seq_len = 2, device = "cuda:0"):
        super().__init__()

        self.seq = []
        self.label = []
        
        for i in range(len(data_sets)):
            if(len(data_sets[i][1]) != seq_len):
                print("Sequence length is wrong. The length must be {} nt".format(seq_len))
            else:
                self.seq.append(data_sets[i][1])
                self.label.append(data_sets[i][2])
    
        print("{} is loaded".format(len(self.label)))
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        token = torch.tensor(numbering(self.seq[idx], mode = "input")).long()
        #emb = torch.nn.functional.one_hot(token, num_classes=len(nu_list["input"]) + 1)
        #emb = emb[None, :, None, :] * emb[:, None, :, None]
        #emb = emb.reshape([emb.size(0), emb.size(1), emb.size(2) * emb.size(3)])
        label = self.label[idx]
        
        return torch.tensor(token).long().to(self.device), torch.tensor(label).float().to(self.device)

def get_pe(seq_lens, max_len, device):
    num_seq = seq_lens.shape[0]
    pos_i_abs = torch.Tensor(np.arange(1,max_len+1)).view(1, -1, 1).expand(num_seq, -1, -1).double().to(device)

    pos_i_rel = torch.Tensor(np.arange(1,max_len+1)).view(1, -1).expand(num_seq, -1).to(device)
    pos_i_rel = pos_i_rel.double()/seq_lens.view(-1, 1).double()
    pos_i_rel = pos_i_rel.unsqueeze(-1)
    pos = torch.cat([pos_i_abs, pos_i_rel], -1)

    PE_element_list = list()
    # 1/x, 1/x^2
    PE_element_list.append(pos)
    PE_element_list.append(1.0/pos_i_abs)
    PE_element_list.append(1.0/torch.pow(pos_i_abs, 2))

    # sin(nx)
    for n in range(1, 50):
        PE_element_list.append(torch.sin(n*pos).to(device))

    # poly
    for i in range(2, 5):
        PE_element_list.append(torch.pow(pos_i_rel, i).to(device))

    for i in range(3):
        gaussian_base = torch.exp(-torch.pow(pos, 2))*math.sqrt(math.pow(2,i)/math.factorial(i))*torch.pow(pos, i)
        PE_element_list.append(gaussian_base.to(device))

    PE = torch.cat(PE_element_list, -1)
    for i in range(num_seq):
        PE[i, seq_lens[i]:, :] = 0
    return PE

def evaluate_exact(pred_a, true_a):
    tp_map = torch.sign(pred_a*true_a)
    tp = tp_map.sum()
    pred_p = torch.sign(pred_a).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = tp/(tp+fn + 1e-6)
    precision = tp/(tp+fp + 1e-6)
    f1_score = 2*tp/(2*tp + fp + fn + 1e-6)
    return precision, recall, f1_score

def f1_loss(pred_a, true_a):
    tp = pred_a*true_a
    tp = torch.sum(tp)

    fp = pred_a*(1-true_a)
    fp = torch.sum(fp)

    fn = (1-pred_a)*true_a
    fn = torch.sum(fn)

    f1 = torch.div(2*tp, (2*tp + fp + fn + 1e-6))
    return 1-f1.mean()

def _make_config(input_dict: dict) -> argparse.Namespace:
    """Recursively go through dictionary"""
    new_dict = {}
    for k, v in input_dict.items():
        if type(v) == dict:
            new_dict[k] = _make_config(v)
        else:
            new_dict[k] = v
    return argparse.Namespace(**new_dict)

def loss_calc(pred, label, weight_val, device):
    weight = torch.zeros_like(label)
    weight[label == 1] = weight_val[1]
    weight[label == 0] = weight_val[0]
    weight = weight.float().to(weight.device)
    criteria = nn.BCELoss(reduction='none')
    loss = criteria(pred, label)
    loss = torch.sum(loss * weight)/torch.sum(weight)

    return loss

class DeepProcess():
    def __init__(self, out_path, train_params, model_cfg, device):
        self.out_path = out_path
        self.train_params = _make_config(train_params)
        self.model_cfg = _make_config(model_cfg)
        self.device = device

    def model_training(self, train_data_sets, val_data_sets):
        os.makedirs(self.out_path, exist_ok=True)

        #print("training data size:: {}".format(len(train_data_sets)), flush = True)
        #print("validation data size:: {}".format(len(val_data_sets)), flush = True)
        
        #トレーニングデータの読み込み
        print("creating dataloader for training...")
        train_sampler = bsampler(train_data_sets, self.train_params.seq_len, device = device)
        train_loader = DataLoader(dataset = train_sampler, batch_size = self.train_params.batch, shuffle=True)
        
        #バリデーションデータの読み込み
        print("creating dataloader for validation...")
        val_sampler = bsampler(val_data_sets, self.train_params.seq_len, device = device)
        val_loader = DataLoader(dataset = val_sampler, batch_size = self.train_params.batch)
        
        #ネットワークモデルの準備
        self.nn_md = Net(self.model_cfg).to(self.device)
        
        #最適化
        self.opt = optim.Adam(params = self.nn_md.parameters(), lr = self.train_params.lr)
        
        #損失計算
        criterion = nn.BCELoss()
        
        nonup_count = 0
        min_loss = 100

        #トレーニング開始
        print("starting to train...")
        for epoch in range(self.train_params.max_epoch):
            print("Epoch_{}=============================".format(epoch + 1), flush = True)
            
            train_pred, val_pred = [], []
            train_label, val_label = [], []
            train_loss, val_loss = [], []
            
            #CNNモデルに入れる
            self.nn_md.train()
            #tokenとlabelにトレーニングセットの配列とラベルを格納する
            #enumerate()→インデックス番号・要素の順に取得
            for i, (token, label) in enumerate(train_loader):
                self.opt.zero_grad()
                
                pe = get_pe(torch.full((token.size(0), 1), self.train_params.seq_len).to(self.device), self.train_params.seq_len, self.device).float().to(self.device)
                
                pred = self.nn_md(token = token, pe = pe).squeeze()
            
                loss = criterion(pred, label)
                print(loss)
                train_pred.extend(pred.cpu().detach().clone().numpy())
                train_label.extend(label.cpu().detach().clone().numpy())
                train_loss.append(loss.item())
                
                loss.backward()
                self.opt.step()
                
            print("training loss:: " + str(np.mean(train_loss)), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_label, train_pred, thresh = 0.5)
                else:
                    metrics = metrics_dict[key](train_label, train_pred)
                print("train_" + key + ": " + str(metrics), flush=True)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_label, train_pred, thresh = 0.5)
            print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
            
            self.nn_md.eval()
            for i, (token, label) in enumerate(val_loader):
                with torch.no_grad():
                    
                    pe = get_pe(torch.full((token.size(0), 1), self.train_params.seq_len).to(self.device), self.train_params.seq_len, self.device).float().to(self.device)
                    
                    pred = self.nn_md(token = token, pe = pe).squeeze()
                
                    loss = criterion(pred, label)
                    val_pred.extend(pred.cpu().detach().clone().numpy())
                    val_label.extend(label.cpu().detach().clone().numpy())
                    val_loss.append(loss.item())
                    
            val_loss = np.mean(val_loss)
            print("validation loss:: "+ str(np.mean(val_loss)), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](val_label, val_pred, thresh = 0.5)
                else:
                    metrics = metrics_dict[key](val_label, val_pred)
                print("validation_" + key + ": " + str(metrics), flush=True)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_label, val_pred, thresh = 0.5)
            print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
            print("", flush=True)

            if(min_loss > val_loss):
                min_loss = val_loss
                os.makedirs(self.out_path + "/data_model", exist_ok=True)
                torch.save(self.nn_md.state_dict(), "{}/data_model/deep_model.pt".format(self.out_path))
                nonup_count = 0
            else:
                nonup_count += 1
                if(nonup_count >= self.train_params.stop_epoch):
                    break

        print("Done!!")

val_size = 0.1
device = "cuda:0"
base_path = "/home/kurata-lab/murata/research-main"  # 自分のディレクトリに合わせる
train_pos_path = base_path + "/dataset/train_set_pos.fa"
train_neg_path = base_path + "/dataset/train_set_neg.fa"
val_pos_path = base_path + "/dataset/val_set_pos.fa"
val_neg_path = base_path + "/dataset/val_set_neg.fa"
out_path = base_path + "/model"
train_pos_data = open_fasta(train_pos_path)
train_neg_data = open_fasta(train_neg_path)
val_pos_data = open_fasta(val_pos_path)
val_neg_data = open_fasta(val_neg_path)

train_pos_data["label"] = 1
train_neg_data["label"] = 0
val_pos_data["label"] = 1
val_neg_data["label"] = 0

train_data = pd.concat([train_pos_data, train_neg_data]).reset_index(drop = True)
val_data = pd.concat([val_pos_data, val_neg_data]).reset_index(drop = True)

train_params = dict(
    lr=0.0001,
    batch=64,
    max_epoch=20000,
    max_seq=1000,
    stop_epoch = 20,
    seq_len = 81
    )

model_cfg = dict(
    dim = 64,
    embed = dict(
        token_size = 6,
        pe_dim = 111
        ),
    encoder = dict(
        depth = 8,
        dim_head = 64,
        heads = 2,
        ff_mult = 4
        )
)

pros = DeepProcess(out_path, train_params, model_cfg, device = device)
temp = pros.model_training(train_data.values.tolist(), val_data.values.tolist())


# temp_1 = temp.detach().clone().numpy()