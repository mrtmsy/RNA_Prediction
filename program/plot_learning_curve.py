#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:13:23 2023

@author: sho
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

def load_data(path):
    with open(path) as f:
        line = f.readline()
        
        data_all = []
        data = []
        while line:
            line = line.replace("\n", "")
            if("Epoch" in line):
                data_all.append(data)
                data = [int(line.split("=")[0].split("_")[-1])]
            elif("loss:: " in line):
                data += [float(line.split("loss:: ")[-1])]
            
            line = f.readline()
        
    return pd.DataFrame(data_all[1:], columns = ["Epoch", "Loss"])


path_pos = "/Users/sho/Documents/support/murata/model/out_pos.log"
path_neg = "/Users/sho/Documents/support/murata/model/out_neg.log"
out_path = "/Users/sho/Documents/support/murata/res"
#data_index = ["loss:: ", "training precision:: ", "training recall:: ", "training f1_score:: ", "validation f1 loss:: ", "validation precision:: ", "validation recall:: ", "validation f1_score:: "]
#data_index_col = ["Epoch", "Training f1 loss", "Training precision", "Training recall", "Training f1 score", "Validation f1 loss", "Validation precision", "Validation recall", "Validation f1 score"]

pos_data = load_data(path_pos).iloc[0:5000, :]
neg_data = load_data(path_neg).iloc[0:5000, :]

plt.style.use('default')
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
#sns.set()
#sns.set_style('whitegrid')
#sns.set_palette('Set1')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(pos_data["Epoch"].values.tolist(), pos_data["Loss"].values.tolist(), label="Positive")
ax.plot(neg_data["Epoch"].values.tolist(), neg_data["Loss"].values.tolist(), label="Negative")
   
plt.legend(ncol=2)
ax.set_xlabel("Epoch", fontsize=15)
ax.set_ylabel("Loss", fontsize=15)
#ax.set_ylim(0.25, 0.75)

plt.savefig("{}/{}".format(out_path, "learning_curve.png"), format="png", dpi=300)




