#positive.faを読み込み、negative.faを生成。
#positive.faとnegative.faをgt_out内に出力。

from Bio import SeqIO
import random
import pandas as pd
import choice
# from sklearn.model_selection import train_test_split

#ファイルからid・seqを抽出し、ラベル化
def load_file(seq_file, label):
  set_data_list = []
  for record in SeqIO.parse(seq_file, 'fasta'):
    desc_part = record.description
    seq_part = record.seq
    set_data_list.append([desc_part, seq_part, label])
    #print(set_data_list)
  return set_data_list

#negative sample 生成
def seq_generation(seq_list):
  for seq in seq_list:
    seq = list(seq)
    n_list = random.sample(seq, len(seq))
    shuffle_list = "".join(n_list)
    # print(shuffle_list + "\n")
    neg_list.append(shuffle_list)
  return neg_list

#重複確認
def check_negative(neg_df, pos_df):
  neg_df[~neg_df["seq"].isin(pos_df["seq"])]  #negとpos比較
  neg_df.drop_duplicates(subset=["seq"], keep=False)  #neg内での比較
  return neg_df

def created_fasta(df, out_file):
  df["id"] = ">" + df["id"]
  df = df.loc[:,["id","seq"]]
  df.to_csv(out_file, sep = '\n', index = None,  header = None)
  return 0

choice.choice()

seq_file = "./positive.fa"
out_pos_fasta = "./gt_out/positive.fa"
out_neg_fasta = "./gt_out/negative.fa"

print("Loading positive sample.")
pos_list = load_file(seq_file, 1)
pos_df = pd.DataFrame(pos_list, columns =["id", "seq", "label"])
seq_list = pos_df["seq"].values.tolist()
print("Loading is completed.")

neg_list = []
neg_df = pd.DataFrame(neg_list, columns =["id", "seq", "label"])
neg_df["id"] = pos_df["id"].str.replace("pos","neg")
neg_df["label"] = 0

print("Negative sample generartion.")
neg_df["seq"] = seq_generation(seq_list)
check_negative(neg_df, pos_df)
print("Negative sample generation is completed.")
# print(neg_df)

# train_pos_data, test_pos_data = train_test_split(pos_df, test_size=0.2)
# train_neg_data, test_neg_data = train_test_split(neg_df, test_size=0.2)
# test_data = pd.concat([test_pos_data,test_neg_data])

# print('Train positive data : ' + str(len(train_pos_data)))
# print('Train negative data : ' + str(len(train_neg_data)))
# print('Test positive data : ' + str(len(test_pos_data)))
# print('Test negative data : ' + str(len(test_neg_data)))

print('positive data : ' + str(len(pos_df)))
print('negative data : ' + str(len(neg_df)))

#out put fasta
# created_fasta(train_pos_data, out_train_pos_fasta)
# created_fasta(train_neg_data, out_train_neg_fasta)
created_fasta(pos_df, out_pos_fasta)
created_fasta(neg_df, out_neg_fasta)
print("Creation of training data fasta file is completed.")
# created_fasta(test_data, out_test_fasta)
# print("Creation of test data fasta file is completed.")

