#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 18:01:36 2023

@author: sho
"""
import os
import pandas as pd
import Bio
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split

def open_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, 'fasta'):
            data.append([record.id, record.seq, record.description])
    return pd.DataFrame(data, columns = ["idx", "seq", "dsp"])

def out_csv(filename, data):
    data.to_csv(filename)

test_size = 0.1
path = "/Users/sho/Documents/support/murata/dataset/train_positive.fa"
out_path = "/Users/sho/Documents/support/murata/dataset"
postive_data = open_fasta("{}/{}".format(path, "train_positive.fa"))
negative_data = open_fasta("{}/{}".format(path, "train_negative.fa"))

data = pd.concat([postive_data, negative_data])
train, test = train_test_split(data, test_size = test_size)

train = train.reset_index(drop = True)
test = test.reset_index(drop = True)

out_csv("{}/{}".format(out_path, "train.csv"), train)
out_csv("{}/{}".format(out_path, "val.csv"), test)


