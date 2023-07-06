#母データから、指定したサンプル数の配列をランダムに抽出し、positive.faを作成する

import random
import pandas as pd
from Bio import SeqIO

#ファイルからid・seqを抽出し、ラベル化
def make_file(seq_file):
  set_data_list =[]
  for record in SeqIO.parse(seq_file, 'fasta'):
    desc_part = record.description
    seq_part = record.seq
    set_data_list.append([desc_part, seq_part])
    #print(set_data_list)
  return set_data_list

def choice():
  in_file = './output.fa'
  out_file= './positive.fa'
  
  seq_count = input('抽出する総データ数を入力(pos+neg)\n')  # 抽出するデータ数 (pos+neg)
  data_pos_list = []
  
  #フォルダ内からディレクトリ名を取得
  data_pos_list.extend(make_file(in_file))
    #positive data listを作成
  # print(data_pos_list)
  data_pos_list = random.sample(data_pos_list, int(seq_count))  #positive data listから抽出
  # print('Positive data'+ str(data_pos_list))
  pos_df = pd.DataFrame(data_pos_list, columns =["id", "seq"])
  pos_df["id"] = ">" + pos_df["id"]
  pos_df.to_csv(out_file, sep = '\n', index = None,  header = None)