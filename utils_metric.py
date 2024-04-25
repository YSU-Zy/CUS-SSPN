import torch
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn
from torch.autograd import Variable
import random
import ot
import numpy as np
from collections import Counter
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, classification_report, precision_score

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

def euclidean_metric_(a,b):
    return torch.norm(a-b,dim=1)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # batch_loss = -alpha*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def CompactLoss(qry_embedding, proto_embedding, labels, focal=False, class_num=None, base_class_num=None):
    dists = euclidean_metric(qry_embedding, proto_embedding)
    if focal:
        alpha = torch.tensor([1.0] * base_class_num + [0.5] * (class_num - base_class_num))
        focal_loss = FocalLoss(class_num, alpha=alpha, gamma=1)
        loss = focal_loss(dists, labels)
    else:
        loss = F.cross_entropy(dists, labels)
    return loss

def UniformityLoss(proto_embedding):
    center_proto_embedding = torch.mean(proto_embedding, dim=0).unsqueeze(0)
    normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding)
    cos_dist_matrix = torch.mm(normalize_proto_embedding, normalize_proto_embedding.T)
    unit_matrix = torch.eye(cos_dist_matrix.shape[0]).cuda()
    cos_dist_matrix = cos_dist_matrix - unit_matrix
    loss = torch.max(cos_dist_matrix, 1).values
    return torch.mean(loss)

def Sparseloss_(proto_embedding):
    normal_proto=F.normalize(proto_embedding)
    center_proto_embedding_ = torch.mean(normal_proto, dim=0).unsqueeze(0)
    dist_matrix_=euclidean_metric(normal_proto,center_proto_embedding_)
    dist_matrix_=torch.exp(torch.min(dist_matrix_,dim=1).values)
    loss=torch.mean(dist_matrix_)
    return loss

def Sparseloss(proto_embedding):
    n_proto=proto_embedding.size(0)
    center_proto_embedding_=proto_embedding.reshape(1,n_proto,-1).mean(dim=1)
    pairwise_distances=torch.cdist(proto_embedding,center_proto_embedding_,p=2)
    loss=torch.tensor(0.0).cuda()
    for i in range(n_proto):
        distance=pairwise_distances[i]
        distance=distance.squeeze()
        loss+=F.relu(distance)
    loss/=(n_proto)
    loss=torch.exp(-loss)
    return loss

def SeparabilityLoss(proto_embedding, teacher_class_num):
    proto_embedding=F.normalize(proto_embedding)
    base_proto = proto_embedding[0: teacher_class_num]
    novel_proto = proto_embedding[teacher_class_num:]
    dist_matrix = euclidean_metric(novel_proto, base_proto)
    min_dist = torch.exp(torch.min(dist_matrix, dim=1).values)
    loss = torch.mean(min_dist)
    return loss

def SeparabilityLoss_(proto_embedding, teacher_class_num,device):
    base_proto = proto_embedding[0: teacher_class_num]
    novel_proto = proto_embedding[teacher_class_num:]
    n_base=base_proto.size(0)
    n_novel=novel_proto.size(0)
    loss=torch.tensor(0.0).to(device)
    pairwise_distances=torch.cdist(novel_proto,base_proto,p=2)
    for i in range(n_novel):
        for j in range(n_base):
            distance=pairwise_distances[i,j]
            distance=distance.squeeze()
            loss+=F.relu(distance-10)
    loss/=(n_novel*n_base)
    loss=torch.exp(-loss)
    return loss

def Knowledge_dist_loss(student_output, teacher_output, T=2.0):
    student_output=F.normalize(student_output)
    teacher_output=F.normalize(teacher_output)
    p = F.softmax(student_output / T, dim=1)
    q = F.softmax(teacher_output/T, dim=1)
    soft_loss = torch.mean(p * torch.log(p/q))
    return soft_loss

def Knowledge_dist_loss_1(student_output, teacher_output, T=2.0):
    student_output=F.normalize(student_output)
    teacher_output=F.normalize(teacher_output)
    p = F.softmax(student_output / T, dim=1)
    q = F.softmax(teacher_output/T, dim=1)
    soft_loss = F.kl_div(torch.log(p),q,reduction='batchmean')*T**2

    return soft_loss

def Knowledge_dist_loss_proto(student_proto, teacher_proto,device):
    n_student_proto=student_proto.size(0)
    loss = torch.tensor(0.0).to(device)
    pairwise_distances=torch.cdist(student_proto,teacher_proto,p=2)
    for i in range(n_student_proto):
        distance=pairwise_distances[i,i].squeeze()
        loss+=distance
    loss/=(n_student_proto)
    return loss

def Knowledge_dist_loss2(student_output, teacher_output, T=2.0):
    student_output=F.normalize(student_output)
    teacher_output=F.normalize(teacher_output)
    p = F.softmax(student_output / T, dim=1)
    q = F.softmax(teacher_output/T, dim=1)
    soft_loss = torch.mean(-torch.sum(q*torch.log(p),dim=1))
    return soft_loss

def spread_loss(proto_embedding):
    num_prototypes=proto_embedding.size(0)
    pairwise_distances=torch.cdist(proto_embedding,proto_embedding,p=2)
    loss=torch.tensor(0.0).cuda()
    for i in range(num_prototypes):
        for j in range(num_prototypes):
            if i!=j:
                distance=pairwise_distances[i,j]
                loss+=F.relu(10-distance)
    loss/=(num_prototypes*(num_prototypes-1))
    return loss

def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity

def loss_unl(train_unlabeled_streamclass_features,train_unlabeled_streamclass_enhance_features,support_streamclass_proto,proto_t,episode,args,class_num):
    if episode<=args.warm_steps:
        logits_unlabeled_streamclass=euclidean_metric(train_unlabeled_streamclass_features,support_streamclass_proto)
        logits_unlabeled_enhance_streamclass=euclidean_metric(train_unlabeled_streamclass_enhance_features,support_streamclass_proto)
        pseu_unlabeled_streamclass=F.softmax(logits_unlabeled_streamclass)
    else:
        logits_unlabeled_streamclass=euclidean_metric(train_unlabeled_streamclass_features,proto_t.mo_pro)
        logits_unlabeled_enhance_streamclass=euclidean_metric(train_unlabeled_streamclass_enhance_features,proto_t.mo_pro)
        pseu_unlabeled_streamclass=F.softmax(logits_unlabeled_streamclass)

    L_intra,pseudo_label_ot=ot_loss(proto_t,train_unlabeled_streamclass_features,train_unlabeled_streamclass_enhance_features,class_num)

    max_probs_unlabeled,target_unlabeled=torch.max(pseu_unlabeled_streamclass,dim=1)

    unl_mask1=max_probs_unlabeled.ge(args.threshold1)
    unl_mask2=max_probs_unlabeled.ge(args.threshold2)

    pseudo_label_ot[unl_mask1]=target_unlabeled[unl_mask1]

    if episode>args.warm_steps:
        unl_pseudo_label=pseudo_label_ot
        unl_mask=unl_mask2.float().unsqueeze(1).detach()
    else:
        unl_pseudo_label=target_unlabeled
        unl_mask=unl_mask1.float().unsqueeze(1).detach()
    L_enhance=1.*(F.cross_entropy(logits_unlabeled_enhance_streamclass,unl_pseudo_label,reduction='none')*unl_mask1).mean()

    return L_intra,L_enhance,unl_mask,unl_pseudo_label



def ot_loss(proto_t, train_unlabeled_streamclass_features, train_unlabeled_streamclass_enhance_features, class_num):
    bs = train_unlabeled_streamclass_enhance_features.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_t.mo_pro, train_unlabeled_streamclass_features)
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    pseudo_label_ot = pred_ot.clone().detach()
    Lm = center_loss_cls(proto_t.mo_pro, train_unlabeled_streamclass_enhance_features, pred_ot, num_classes=class_num)
    return Lm, pseudo_label_ot

def ot_mapping(M):
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma

def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat

def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat =  - 1. * x @ centers_norm.t() + 1
    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))
    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss

