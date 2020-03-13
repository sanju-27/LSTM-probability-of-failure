'''
Created on 02 Mar 2019

@author: sanju-27


Use Trained LSTM model for Predictive Maintenance of a Machine

Predict the Probability that a machine will fail in var: period time period

Insert the predicted value to database

'''

import keras
import argparse
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import gen_sequence, gen_label
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import mysql.connector

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   passwd="123",
#   database="test"
# )
# mycursor = mydb.cursor()
parser = argparse.ArgumentParser()
parser.add_argument("machine", help="machine id",type=int)
args = parser.parse_args()

dataset_train=pd.read_csv('./dataset/PM_train.txt',sep=' ',header=None).drop([26,27],axis=1)
col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
dataset_train.columns=col_names
# print('Shape of Train dataset: ',dataset_train.shape)
# dataset_train.head()
dataset_test=pd.read_csv('./dataset/PM_test.txt',sep=' ',header=None).drop([26,27],axis=1)
dataset_test.columns=col_names
#dataset_test.head()
# print('Shape of Test dataset: ',dataset_train.shape)
# dataset_test.head()

pm_truth=pd.read_csv('./dataset/PM_truth.txt',sep=' ',header=None).drop([1],axis=1)
pm_truth.columns=['more']
pm_truth['id']=pm_truth.index+1
# pm_truth.head()
rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
# rul.head()

pm_truth['rtf']=pm_truth['more']+rul['max']
# pm_truth.head()
pm_truth.drop('more', axis=1, inplace=True)
dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')
dataset_test['ttf']=dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)
# dataset_test.head()
dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']
# dataset_train.head()
df_train=dataset_train.copy()
df_test=dataset_test.copy()
period=30
df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)
# df_train.head()
features_col_name=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                   's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name='label_bc'
sc=MinMaxScaler()
df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
df_test[features_col_name]=sc.transform(df_test[features_col_name])

seq_length=50
seq_cols=features_col_name
X_train=np.concatenate(list(list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols)) for id in df_train['id'].unique()))
# print(X_train.shape)
# generate y_train
y_train=np.concatenate(list(list(gen_label(df_train[df_train['id']==id], 50, seq_cols,'label_bc')) for id in df_train['id'].unique()))
# print(y_train.shape)
X_test=np.concatenate(list(list(gen_sequence(df_test[df_test['id']==id], seq_length, seq_cols)) for id in df_test['id'].unique()))
# print(X_test.shape)
# generate y_test
y_test=np.concatenate(list(list(gen_label(df_test[df_test['id']==id], 50, seq_cols,'label_bc')) for id in df_test['id'].unique()))
# print(y_test.shape)
nb_features =X_train.shape[2]
timestamp=seq_length

model = keras.models.load_model('my_model.h5')
# model.summary()

def prob_failure(machine_id):
    machine_df=df_test[df_test.id==machine_id]
    machine_test=gen_sequence(machine_df,seq_length,seq_cols)
    m_pred=model.predict(machine_test)
    failure_prob=list(m_pred[-1]*100)[0]
    return failure_prob

machine_id=args.machine

# machine_id = 1
ppp = str(prob_failure(machine_id))
print('Probability that machine',machine_id, 'will fail within 30 days: ',ppp)

# sql = "INSERT INTO prob (m_id, probability) VALUES (%s, %s)"
# val = (machine_id, ppp)
# mycursor.execute(sql, val)

# mydb.commit()

# print(mycursor.rowcount, "record inserted.")

# machine_id = 2
# ppp = str(prob_failure(machine_id))
# print('Probability that machine',machine_id, 'will fail within 30 days: ',ppp)

# sql = "INSERT INTO prob (m_id, probability) VALUES (%s, %s)"
# val = (machine_id, ppp)
# mycursor.execute(sql, val)

# mydb.commit()

# print(mycursor.rowcount, "record inserted.")