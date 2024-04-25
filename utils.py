import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import random
import time
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.metrics import confusion_matrix
import math
import warnings
warnings.filterwarnings("ignore")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


def radiation_noise(data, alpha_range=(0.99, 1.01), beta=1/100):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def radiation_noise_enhance(data, alpha_range=(0.98, 1.02), beta=1/80):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

class Task(object):
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)

class Task1(object):
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        #class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_folders)))

        labels = dict(zip(class_folders, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_folders:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader


class Prototype_t:
    def __init__(self, C=65, dim=512, m=0.5):  # 接收三个参数：C表示类别数（类别数量），dim表示原型向量的维度，m表示动量参数（momentum）。
        self.mo_pro = torch.zeros(C, dim).cuda()  # 在初始化过程中，创建了两个存储原型向量的张量mo_pro和batch_pro，形状为(C, dim)，并将它们移动到GPU上。
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m

    @torch.no_grad()
    def update(self, support_streamclass_proto,support_streamclass_features,support_streamclass_label,train_unlabeled_streamclass_features, episode, unl_mask, unl_pseudo_label, args,class_num,device,j,
               norm=False):
        if episode < 20:
            momentum = 0
        else:
            momentum = self.m

        if episode <= (20 + args.warm_steps) / 2:
            for class_ in range(class_num):
                if class_==(args.stream_n_base+j):
                    self.mo_pro[class_,:]=self.mo_pro[class_,:]*momentum+support_streamclass_proto[class_,:]*(1-momentum)
                else:
                    self.mo_pro[class_,:]=support_streamclass_proto[class_,:]

        # unlabel update
        if episode > (20 + args.warm_steps) / 2:
            lblt=support_streamclass_label[(class_num-1)*args.K_shot:].to(device)
            featt=support_streamclass_features[(class_num-1)*args.K_shot:,:].to(device)
            unl_pseudo_label1 = torch.cat([lblt, unl_pseudo_label], dim=0)
            feat_tu_w_for_pro1 = torch.cat([featt, train_unlabeled_streamclass_features], dim=0)
            unl_mask1 = torch.cat([torch.ones(lblt.shape[0], 1).cuda(), unl_mask], dim=0)
            for class_ in range(class_num):
                if class_>=(class_num-args.stream_n_novel_list[j]):
                    featt_i = feat_tu_w_for_pro1[unl_pseudo_label1 == class_ * unl_mask1.squeeze(), :]
                    if featt_i.shape[0]:
                        featt_i_center = featt_i.mean(dim=0, keepdim=True)
                        self.mo_pro[class_, :] = self.mo_pro[class_, :] * momentum + featt_i_center * (1 - momentum)
                else:
                    self.mo_pro[class_,:]=support_streamclass_proto[class_,:]

        if norm:
            self.mo_pro = F.normalize(self.mo_pro)

