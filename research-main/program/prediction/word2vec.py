import os
import pandas as pd
import Bio
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
from gensim.models import word2vec
import logging
from gensim.models import KeyedVectors


def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
        data.append([record.id, str(record.seq[0:81])])
    return pd.DataFrame(data, columns=["id", "seq"])

def word_list(seqs):
  total_list = []
  length = 4
  for seq in seqs:
  #n_listに78*4の4-merを格納
    n_list = []
    for i in range(len(seq)-length + 1):
      n_list.append(str(seq[i:i+length]))
    total_list.append(n_list)
  return total_list

def model_w2v(seqs, out_path):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  model = word2vec.Word2Vec(sentences=seqs, vector_size=64, window=3, min_count=0, epochs = 100)
  os.makedirs(out_path, exist_ok = True)
  model.wv.save_word2vec_format(out_path + "/w2v_model.pt", binary=True)

def encode(seqs,out_path):
  model = KeyedVectors.load_word2vec_format(out_path + "/w2v_model.pt", binary=True)
  total_list = []
  for seq in seqs:
    vec_list = []
    for word in seq:
      vec_list.append(np.array(model[word]))
  total_list.append(vec_list)
  return total_list


base_path = "/home/kurata-lab/murata/research-main"  # 自分のディレクトリに合わせる
out_path = base_path + "./w2v_model"
train_pos_path = base_path + "/dataset/train_set_pos.fa"
train_neg_path = base_path + "/dataset/train_set_neg.fa"
val_pos_path = base_path + "/dataset/val_set_pos.fa"
val_neg_path = base_path + "/dataset/val_set_neg.fa"

#データ読み込み
train_pos_data = open_fasta(train_pos_path)
val_pos_data = open_fasta(val_pos_path)
# train_neg_data = open_fasta(train_neg_path)
# val_neg_data = open_fasta(val_neg_path)
# print(train_pos_data)

#各配列を4つに区切る
train_pos_list = word_list(train_pos_data["seq"])
val_pos_list = word_list(val_pos_data["seq"])
# train_neg_list = word_list(train_neg_data["seq"])
# val_neg_list = word_list(val_neg_data["seq"])

#モデル構築
model_w2v(train_pos_list + val_pos_list, out_path)

#エンコード
# train_pos_list = encode(train_pos_list,out_path)
# val_pos_list = encode(val_pos_list, out_path)

# # train_neg_list = encode(train_neg_list, out_path)
# # val_neg_list = encode(val_neg_list, out_path)
# # print(train_pos_list)

# train_pos_list = np.array(train_pos_list)
# val_pos_list = np.array(val_pos_list)

# train_neg_list = np.array(train_neg_list)
# val_neg_list = np.array(val_neg_list)