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

def encode(seqs,out_path):
  model = KeyedVectors.load_word2vec_format(out_path + "/w2v_model.pt", binary=True)
  length = 4
  n_list = []
  vec_list = []
  for i in range(len(seqs)-length + 1):
    n_list.append(str(seqs[i:i+length]))
  vec_list.append(np.array(model[n_list]))
  return vec_list


def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
        data.append([record.id, str(record.seq[0:81])])
    return pd.DataFrame(data, columns=["id", "seq"])