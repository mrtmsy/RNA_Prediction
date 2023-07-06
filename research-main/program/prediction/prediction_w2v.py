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
from model_attn_w2v import Net
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}
import warnings
warnings.filterwarnings('ignore')
import w2v_encoder

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
            #if(len(data_sets[i][1]) != seq_len):
            #print("Sequence length is wrong. The length must be {} nt".format(seq_len))
            #else:
            self.seq.append(data_sets[i][1])
            self.label.append(data_sets[i][2])
    
        print("{} is loaded".format(len(self.label)))
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        token = w2v_encoder.encode(self.seq[idx], w2v_out_path)
        # token = torch.tensor(numbering(self.seq[idx], mode = "input")).long()
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
    def __init__(self, model_path, params, model_cfg, device):
        self.model_path = model_path
        self.params = _make_config(params)
        self.model_cfg = _make_config(model_cfg)
        self.device = device
        
    def model_prediction(self, data_sets):
        #os.makedirs(self.out_path, exist_ok=True)

        #print("training data size:: {}".format(len(train_data_sets)), flush = True)
        #print("validation data size:: {}".format(len(val_data_sets)), flush = True)
        
        print("creating dataloader for evaluation...")
        sampler = bsampler(data_sets, self.params.seq_len, device = device)
        loader = DataLoader(dataset = sampler, batch_size = self.params.batch)

        self.nn_md = Net(self.model_cfg).to(self.device)
        self.nn_md.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        
        pred_list, label_list, loss_list = [], [], []
            
        criterion = nn.BCELoss()
        
        
        self.nn_md.eval()
        
        tim = []
        for i, (token, label) in enumerate(loader):
            pe = get_pe(torch.full((token.size(0), 1), self.params.seq_len).to(self.device), self.params.seq_len, self.device).float().to(self.device)
            
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                pred = self.nn_md(token = token, pe = pe).squeeze()
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
            tim.append(elapsed_time/1000)
            
            loss = criterion(pred, label)
            pred_list.extend(pred.cpu().detach().clone().numpy())
            label_list.extend(label.cpu().detach().clone().numpy())
            loss_list.append(loss.item())
        
        print("loss:: " + str(np.mean(loss_list)), flush = True)
        print(len(tim))
        print("Run time:: {} sec".format(sum(tim)/len(data_sets)))

        for key in metrics_dict.keys():
            if(key != "auc" and key != "AUPRC"):
                metrics = metrics_dict[key](label_list, pred_list, thresh = 0.5)
            else:
                metrics = metrics_dict[key](label_list, pred_list)
            print("test_" + key + ": " + str(metrics), flush=True)
        
        tn_t, fp_t, fn_t, tp_t = cofusion_matrix(label_list, pred_list, thresh = 0.5)
        print("test_true_negative:: value: %f" % (tn_t), flush=True)
        print("test_false_positive:: value: %f" % (fp_t), flush=True)
        print("test_false_negative:: value: %f" % (fn_t), flush=True)
        print("test_true_positive:: value: %f" % (tp_t), flush=True)

        print("Done!!")


val_size = 0.1
device = "cuda:0"
base_path = "/home/kurata-lab/murata/research-main"
pos_path = base_path + "/dataset/test_positive.fa"
neg_path = base_path + "/dataset/test_negative.fa"
model_path = base_path + "/model/w2v_deep_model.pt"
w2v_out_path = base_path + "/w2v_model"
pos_data = open_fasta(pos_path)
neg_data = open_fasta(neg_path)

pos_data["label"] = 1
neg_data["label"] = 0

data = pd.concat([pos_data, neg_data]).reset_index(drop = True)

params = dict(
    batch=64,
    seq_len = 78
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

pros = DeepProcess(model_path, params, model_cfg, device = device)
pros.model_prediction(data.values.tolist())
    
    
