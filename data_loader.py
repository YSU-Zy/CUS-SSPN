from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def standardization(data):
    data = data.T
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data.T

class DiabetesDataset(Dataset):
    def __init__(self,data, label):
        # xy = np.loadtxt(filepath, delimiter =',', dtype =np.float32)
        self.len = data.shape[0]
        data = data.astype(np.float32)
        self.x_data = torch.from_numpy(data).view(-1,1,data.shape[1])  #(-1,通道,行，列)
        self.y_data = torch.from_numpy(label).view(-1).long()
        # print(self.x_data.size())
        # print(self.y_data.size())
    def __getitem__(self, item):
        return self .x_data[item], self .y_data[item]
    def __len__(self):
        return self.len

def load_data(data, label,batch_size=32):
    dataset = DiabetesDataset(data, label)
    data_loader = DataLoader(dataset =dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False)
    return data_loader

def load_data_1(data, label,batch_size=32):
    dataset = DiabetesDataset(data, label)
    data_loader_1 = DataLoader(dataset =dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)
    return data_loader_1

def load_data_unlabeled(data,label,batch_size=32):
    dataset=DiabetesDataset(data,label)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    return data_loader