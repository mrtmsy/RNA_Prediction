import os
import pandas as pd
import Bio
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
from sklearn.model_selection import train_test_split


def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
        data.append([record.id, record.seq[0:81]])
    return pd.DataFrame(data, columns=["id", "seq"])

def out_fasta(filename, df):
  df['id'] = '>' + df['id'].astype(str)
  df.to_csv(filename, sep='\n', index=None,  header=None)


base_path = "/home/kurata-lab/murata/research-main"  # 自分のディレクトリに合わせる
train_pos_path = base_path + "/dataset/train_positive.fa"
train_neg_path = base_path + "/dataset/train_negative.fa"
out_train_pos_f = base_path + "/dataset/train_set_pos.fa"
out_val_pos_f = base_path + "/dataset/val_set_pos.fa"
out_train_neg_f = base_path + "/dataset/train_set_neg.fa"
out_val_neg_f = base_path + "/dataset/val_set_neg.fa"

print("loading train file...")
train_pos_data = open_fasta(train_pos_path)
train_neg_data = open_fasta(train_neg_path)

print("separating data...")
train_pos_data, val_pos_data, train_neg_data, val_neg_data = train_test_split(train_pos_data, train_neg_data, test_size=0.1, random_state=0)

print("output file...")
out_fasta(out_train_pos_f,train_pos_data)
out_fasta(out_train_neg_f,train_neg_data)
out_fasta(out_val_pos_f,val_pos_data)
out_fasta(out_val_neg_f,val_neg_data)