import pandas as pd
import os
import numpy as np
from scipy.signal import argrelextrema
import scipy.interpolate as spi
def data_process(data_num,data_len):
    '''
    data_num:Number of samples per category
    data_len：Length of each sample
    '''
    n=0
    data_,label_=[],[]
    for i in os.listdir('Data/***'): #Input dataset path
        if n<7:
            print(i,',为第',n,'类')
            # file=pd.read_excel('Data/N0-H10-W1-20S-2.xlsx').iloc[:,:6].values
            file=pd.read_excel('Data/***/'+i,engine='openpyxl').iloc[:200000,:1].values
            for j in range(data_num):
                start=np.random.randint(0,file.shape[0]-data_len)  #np.random.randint()返回随机整数
                #  np.random.randint（low，high，size，dtype）；生成随机整数范围【low，high）
                end=start+data_len
                data_.append(file[start:end,:])
                label_.append(n)
            n += 1
    data_=np.array(data_)
    label_=np.array(label_)
    return data_,label_
num=500
N=2048
data_,label_=data_process(num,N)
'''划分后数据为:****x2048x1'''
# from scipy.fftpack import fft
#
# fea=[]
# for i in range(data_.shape[0]):
#     fea1=np.zeros([1024,6])
#     for j in range(data_.shape[2]):
#         fft_y=fft(data_[i,:,j])
#         abs_y=np.abs(fft_y)
#         normalization_y=abs_y/N
#         tz = normalization_y[range(int(N/2))]
#         fea1[:,j]=tz
#     fea.append(fea1)
# fea=np.array(fea)
# '''划分后数据为:2000x1024x1'''

# In[]保存数据
np.savez('result/***/***.npz', data=data_, label=label_)
#np.savez('result/RM_stream/RM_4_base_5idx_1_novel_5_shot_stream.npz', data=data_, label=label_)
# np.savez('result/target_data_feature_paderborn_200.npz', data=fea, label=label_)









