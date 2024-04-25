import os
import time
import math
import torch
import random
import warnings
import argparse
import openpyxl
import encoder_model
import numpy as np
from utils import *
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from utils_metric import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler,MinMaxScaler

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CSU-SSPN')
parser.add_argument('--model',default='CSU_SSPN',type=str,help='Proposed model')
parser.add_argument('--stream_name', default='***', type=str,help='Enter the name of the experimental dataset')
parser.add_argument('--data_channel',default=0,type=int,help='Data channel for experimental dataset')
parser.add_argument('--stream_n_base',default=3,type=int,help='Number of base classes')
parser.add_argument('--stream_n_novel_list',default=[2,2,1,1,1],type=int,help='Novel stream list')
parser.add_argument('--iteration',default=10,type=int,help='Number of repeated experiments')
parser.add_argument('--episodes',default=400,type=int,help='Number of episodes of base classes training')
parser.add_argument('--episodes_stream',default=1500,type=int,help='Number of episodes of stream training')
parser.add_argument('--lr',default=0.001,type=int,help='Learning rate of base classes training')
parser.add_argument('--lr_stream',default=0.0001,type=int,help='Learning rate of stream training')
parser.add_argument('--weight_decay',default=5e-4,type=float,help='Weight decay of optimizer')
parser.add_argument('--K_shot',default=5,type=int,help='The values of K')
parser.add_argument('--tf',default=True,type=bool,help='Whether to save log_dir')
parser.add_argument('--feature_dim',default=1536,type=int,help='Number of feature dimensions ')
parser.add_argument('--query_num_per_class',default=15,type=int,help='Number of query samples per class')
parser.add_argument('--unlabeled_streamclass_samples_num',default=20,type=int,help='Number of unlabeled samples per class in stream training')
parser.add_argument('--warm_steps',default=500,type=int,help='Start of prototype undate')
parser.add_argument('--threshold1',default=0.95,type=float,help='pseudo label threshold1')
parser.add_argument('--threshold2',default=0.8,type=float,help='pseudo label threshold2')
parser.add_argument('--test_lsamples_num_per_class_base',default=100,type=int,help='Number of labeled samples per class in base training')
parser.add_argument('--test_lsamples_num_per_class_stream',default=10,type=int,help='Number of labeled samples per class in stream training')
args = parser.parse_args()



if __name__=='__main__':

    if torch.cuda.is_available():
        device=torch.device('cuda')
        print('Device:GPU')
    else:
        device=torch.device('cpu')
        print('Device:CPU')

    total_accuracy_test=[]
    total_accuracy_list=[]

    processed_data_path_prefix='result/{}_stream/'.format(args.stream_name)
    pkl_path='model_pkl/{}/'.format(args.stream_name)
    result_path='diag_result/{}/'.format(args.stream_name)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.tf==True:
        writer=SummaryWriter('log_dir')

    AVG_baseclass_train = []
    AVG_baseclass_test=[]
    for i in range(args.iteration):
        acc_list=[]
        test_acc_list=[]
        pkl_filename=pkl_path+"{}_{}_{}.pkl".format(args.stream_name,args.model,i)
        proto_pkl_filename=pkl_path+"{}_{}_proto_{}.pkl".format(args.stream_name,args.model,i)
        prediction_filename=result_path+"{}_{}_prediction_{}.npy".format(args.stream_name,args.model,i)
        start_time = time.time()
        print("Iteration of experiment: [{}/{}]".format(i+1, args.iteration))

        base_npz_filename="{}_{}_base_stream.npz".format(args.stream_name,args.stream_n_base)
        baseclass_information=np.load(processed_data_path_prefix+base_npz_filename)
        train_baseclass_loader,test_baseclass_loader,meat_train_baseclass=baseclass_data_process(baseclass_information,args,i)

        feature_extractor=encoder_model.CNN(feature_dim=args.feature_dim)
        feature_extractor.apply(weights_init)
        feature_extractor.to(device)
        feature_extractor_optimizer=torch.optim.Adam(feature_extractor.parameters(),lr=args.lr,weight_decay=args.weight_decay)

        base_class_num=args.stream_n_base
        class_num=args.stream_n_base
        n_novel_list=args.stream_n_novel_list
        best_base_acc=0
        train_baseclass_iter=iter(train_baseclass_loader)

        for episode in tqdm(range(args.episodes)):
            try:
                train_baseclass_data,train_baseclass_label=train_baseclass_iter.next()
            except Exception as err:
                train_baseclass_iter=iter(train_baseclass_loader)
                train_baseclass_data, train_baseclass_label = train_baseclass_iter.next()
            feature_extractor.train()
            feature_extractor_optimizer.zero_grad()
            task_baseclass=Task(meat_train_baseclass,args.stream_n_base,args.K_shot,args.query_num_per_class)
            support_dataloader_baseclass=get_HBKC_data_loader(task_baseclass,num_per_class=args.K_shot,split="train",shuffle=False)
            query_dataloader_baseclass=get_HBKC_data_loader(task_baseclass,num_per_class=args.query_num_per_class,split="test",shuffle=False)
            support_baseclass_data,support_baseclass_label=support_dataloader_baseclass.__iter__().next()
            query_baseclass_data,query_baseclass_label=query_dataloader_baseclass.__iter__().next()
            support_baseclass_data=support_baseclass_data.unsqueeze(1).type(torch.FloatTensor)
            query_baseclass_data=query_baseclass_data.unsqueeze(1).type(torch.FloatTensor)
            support_baseclass_features,support_baseclass_outputs=feature_extractor(support_baseclass_data.to(device))
            query_baseclass_features,query_baseclass_outputs=feature_extractor(query_baseclass_data.to(device))
            if args.K_shot>1:
                support_baseclass_proto=support_baseclass_features.reshape(args.stream_n_base,args.K_shot,-1).mean(dim=1)
            else:
                support_baseclass_proto=support_baseclass_features
            query_baseclass_label=torch.as_tensor(query_baseclass_label,dtype=torch.long).to(device)
            logits_baseclass=euclidean_metric(query_baseclass_features,support_baseclass_proto)
            c_loss=CompactLoss(query_baseclass_features,support_baseclass_proto,query_baseclass_label)
            u_loss=UniformityLoss(support_baseclass_proto)
            s_loss=Sparseloss(support_baseclass_proto)
            Total_loss_baseclass=c_loss+u_loss+s_loss
            Total_loss_baseclass.backward()
            feature_extractor_optimizer.step()
            hit_baseclass=torch.sum(torch.argmax(logits_baseclass,dim=1).to(device)==query_baseclass_label).item()
            num_baseclass=query_baseclass_data.shape[0]
            acc_train=hit_baseclass/num_baseclass

            if episode % 10==0:
                feature_extractor.eval()
                feature_test_baseclass=torch.tensor([])
                label_test_baseclass=torch.tensor([])
                with torch.no_grad():
                    train_features_baseclass,_=feature_extractor(Variable(train_baseclass_data).to(device))
                    if args.test_lsamples_num_per_class_base>1:
                        support_baseclass_proto_test=train_features_baseclass.reshape(args.stream_n_base,args.test_lsamples_num_per_class_base,-1).mean(dim=1)
                    else:
                        support_baseclass_proto_test=train_features_baseclass
                    for test_baseclass_data,test_baseclass_label in test_baseclass_loader:
                        test_baseclass_features,test_baseclass_outputs=feature_extractor(Variable(test_baseclass_data).to(device))
                        feature_test_baseclass=torch.cat((feature_test_baseclass.to(device),test_baseclass_features.to(device)),dim=0)
                        label_test_baseclass=torch.cat((label_test_baseclass,test_baseclass_label),dim=0)

                label_test_baseclass=torch.as_tensor(label_test_baseclass,dtype=torch.long)
                test_logits_baseclass=euclidean_metric(feature_test_baseclass,support_baseclass_proto_test)
                total_rewards=torch.sum(torch.argmax(test_logits_baseclass,dim=1).to(device)==label_test_baseclass.to(device)).item()
                test_acc=100.*total_rewards/len(test_baseclass_loader.dataset)
                AVG_baseclass_test.append(test_acc)

                if test_acc>best_base_acc:
                    best_base_acc=test_acc
                    torch.save(feature_extractor.state_dict(),pkl_filename)
                    best_episode=episode
                    test_proto=support_baseclass_proto_test
                    test_query=feature_test_baseclass
                    test_gt=label_test_baseclass

        AVG_baseclass_train.append(best_base_acc)
        print("[Base training] Best acc: {}% @episode {} test_data_len={}".format(best_base_acc, best_episode,len(test_baseclass_loader.dataset)))
        acc_list.append(best_base_acc)
        proto_embedding_filename=result_path+"{}_proto_embedding_stream{}.npy".format(args.model,0)
        query_embedding_filename=result_path+"{}_query_embedding_stream{}.npy".format(args.model,0)
        ground_truth_filename=result_path+"{}_ground_truth_stream{}.npy".format(args.model,0)
        np.save(proto_embedding_filename,test_proto.detach().cpu().numpy())
        np.save(query_embedding_filename,test_query.detach().cpu().numpy())
        np.save(ground_truth_filename,test_gt.detach().cpu().numpy())

        del support_baseclass_data, support_baseclass_label, query_baseclass_data, query_baseclass_label, train_baseclass_data, train_baseclass_label, test_baseclass_data, test_baseclass_label
        del feature_extractor

        for j in range(0,len(args.stream_n_novel_list)):
            proto_embedding_filename=result_path+"{}_proto_embedding_stream{}.npy".format(args.model,j+1)
            query_embedding_filename = result_path + "{}_query_embedding_stream{}.npy".format(args.model, j+1)
            ground_truth_filename = result_path + "{}_ground_truth_stream{}.npy".format(args.model, j+1)
            supplement_data_filename='result/supplement_data_streamclass{}.npz'.format(j)

            print("===============Streaming {} ===============".format(j+1))
            ft_feature_extractor=encoder_model.CNN(feature_dim=args.feature_dim)
            ft_feature_extractor.load_state_dict(torch.load(pkl_filename))
            ft_feature_extractor.to(device)
            ft_feature_extractor_optimizer=torch.optim.Adam(ft_feature_extractor.parameters(),lr=args.lr_stream,weight_decay=args.weight_decay)

            teacher_feature_extractor=encoder_model.CNN(feature_dim=args.feature_dim)
            teacher_feature_extractor.load_state_dict(torch.load(pkl_filename))
            teacher_feature_extractor.to(device)
            teacher_feature_extractor.eval()

            print("Novel class in streaming......")
            teacher_class_num=class_num
            class_num=class_num+n_novel_list[j]
            print("Teacher class num is {}".format(teacher_class_num))
            best_ft_test_acc=0

            stream_npz_iter_filename="{}_{}_base_{}idx_{}_novel_{}_shot_stream.npz".format(args.stream_name,args.stream_n_base,j,n_novel_list[j],args.K_shot)
            stream_information=np.load(processed_data_path_prefix+stream_npz_iter_filename)
            if j>=1:
                train_streamclass_loader,test_streamclass_loader,train_unlabeled_streamclass_loader,train_unlabeled_streamclass_enhance_loader,meat_train_streamclass=stream_data_process_K_shot(stream_information,args,i,j,class_num)
            else:
                train_streamclass_loader, test_streamclass_loader, train_unlabeled_streamclass_loader, train_unlabeled_streamclass_enhance_loader, meat_train_streamclass = stream_data_process_K_shot(stream_information, args, i, j,class_num)

            train_streamclass_iter = iter(train_streamclass_loader)
            train_unlabeled_streamclass_iter = iter(train_unlabeled_streamclass_loader)
            train_unlabeled_streamclass_enhance_iter = iter(train_unlabeled_streamclass_enhance_loader)

            proto_t = Prototype_t(C=class_num, dim=args.feature_dim)
            supplement_data=torch.tensor([])

            for episode in range(args.episodes_stream):
                try:
                    train_streamclass_data,train_streamclass_label=train_streamclass_iter.next()
                except Exception as err:
                    train_streamclass_iter = iter(train_streamclass_loader)
                    train_streamclass_data, train_streamclass_label = train_streamclass_iter.next()
                try:
                    train_unlabeled_streamclass_data, train_unlabeled_streamclass_label = train_unlabeled_streamclass_iter.next()
                except Exception as err:
                    train_unlabeled_streamclass_iter = iter(train_unlabeled_streamclass_loader)
                    train_unlabeled_streamclass_data, train_unlabeled_streamclass_label = train_unlabeled_streamclass_iter.next()
                try:
                    train_unlabeled_streamclass_enhance_data, _ = train_unlabeled_streamclass_enhance_iter.next()
                except Exception as err:
                    train_unlabeled_streamclass_enhance_iter = iter(train_unlabeled_streamclass_enhance_loader)
                    train_unlabeled_streamclass_enhance_data, _ = train_unlabeled_streamclass_enhance_iter.next()

                ft_feature_extractor.train()
                ft_feature_extractor_optimizer.zero_grad()
                task_streamclass=Task1(meat_train_streamclass,class_num,args.K_shot,args.query_num_per_class)
                support_dataloader_streamclass=get_HBKC_data_loader(task_streamclass,num_per_class=args.K_shot,split="train",shuffle=False)
                query_dataloader_streamclass=get_HBKC_data_loader(task_streamclass,num_per_class=args.query_num_per_class,split="test",shuffle=False)
                support_streamclass_data,support_streamclass_label=support_dataloader_streamclass.__iter__().next()
                query_streamclass_data,query_streamclass_label=query_dataloader_streamclass.__iter__().next()
                support_streamclass_data=support_streamclass_data.unsqueeze(1).type(torch.FloatTensor)
                query_streamclass_data=query_streamclass_data.unsqueeze(1).type(torch.FloatTensor)
                support_streamclass_features,support_streamclass_outputs=ft_feature_extractor(support_streamclass_data.to(device))
                query_streamclass_features,query_streamclass_outputs=ft_feature_extractor(query_streamclass_data.to(device))

                train_unlabeled_streamclass_features, _ = ft_feature_extractor(Variable(train_unlabeled_streamclass_data).to(device))
                train_unlabeled_streamclass_enhance_features, _ = ft_feature_extractor(Variable(train_unlabeled_streamclass_enhance_data).to(device))

                support_streamclass_features_teacher,support_streamclass_outputs_teacher=teacher_feature_extractor(support_streamclass_data.to(device))
                query_streamclass_features_teacher,query_streamclass_outputs_teacher=teacher_feature_extractor(query_streamclass_data.to(device))

                if args.K_shot>1:
                    support_streamclass_proto=support_streamclass_features.reshape(class_num,args.K_shot,-1).mean(dim=1)
                    support_streamclass_proto_teacher=support_streamclass_features_teacher.reshape(class_num,args.K_shot,-1).mean(dim=1)
                else:
                    support_streamclass_proto=support_streamclass_features
                    support_streamclass_proto_teacher=support_streamclass_features_teacher

                L_intra, L_enhance, unl_mask, unl_pseudo_label = loss_unl(train_unlabeled_streamclass_features,train_unlabeled_streamclass_enhance_features,
                                                                          support_streamclass_proto, proto_t, episode,args, class_num)

                proto_t.update(support_streamclass_proto, support_streamclass_features, support_streamclass_label,train_unlabeled_streamclass_features, episode, unl_mask, unl_pseudo_label, args, class_num, device, j, norm=False)

                query_streamclass_label=torch.as_tensor(query_streamclass_label,dtype=torch.long).to(device)
                teacher_logits = euclidean_metric(query_streamclass_features_teacher, support_streamclass_proto_teacher)

                if episode<=args.warm_steps:
                    logits_streamclass=euclidean_metric(query_streamclass_features,support_streamclass_proto)
                    c_loss=CompactLoss(query_streamclass_features,support_streamclass_proto,query_streamclass_label)
                    u_loss=UniformityLoss(support_streamclass_proto)
                    s_loss=Sparseloss(support_streamclass_proto)
                    m_loss=Knowledge_dist_loss(logits_streamclass[:,0:teacher_class_num],teacher_logits[:,0:teacher_class_num],T=2)
                    m_loss_proto=Knowledge_dist_loss_proto(support_streamclass_proto[0:teacher_class_num],support_streamclass_proto_teacher[0:teacher_class_num],device)
                    Total_loss_streamclass=0.6*(c_loss+u_loss+s_loss)+0.4*(m_loss_proto+m_loss)

                else:
                    logits_streamclass = euclidean_metric(query_streamclass_features, proto_t.mo_pro)
                    c_loss=CompactLoss(query_streamclass_features, proto_t.mo_pro,query_streamclass_label)
                    u_loss = UniformityLoss(proto_t.mo_pro)
                    s_loss = Sparseloss(proto_t.mo_pro)
                    m_loss = Knowledge_dist_loss(logits_streamclass[:, 0:teacher_class_num],teacher_logits[:, 0:teacher_class_num], T=2)
                    m_loss_proto = Knowledge_dist_loss_proto(proto_t.mo_pro[0:teacher_class_num],support_streamclass_proto_teacher[0:teacher_class_num],device)
                    Total_loss_streamclass = 0.6*(c_loss+u_loss+s_loss)+0.4*(m_loss_proto+m_loss)
                    if episode>(args.warm_steps+50) and episode<(args.warm_steps+53):
                        for iii in unl_mask:
                            if iii==1:
                                supplement_data=torch.cat([supplement_data,train_unlabeled_streamclass_data[torch.as_tensor(iii,dtype=torch.long)]],dim=0)

                if args.tf==True:
                    writer.add_scalar('Total_loss_streamclass_'+str(j),Total_loss_streamclass,episode)
                    writer.add_scalar('Compact_loss_'+str(j),c_loss,episode)
                    writer.add_scalar('Uniformity_loss_' + str(j), u_loss, episode)
                    writer.add_scalar('Sparse_loss_'+str(j),s_loss,episode)
                    writer.add_scalar('Knowledge_dist_loss_' + str(j), m_loss, episode)
                    writer.add_scalar('Knowledge_dist_loss_proto_' + str(j), m_loss_proto, episode)

                Total_loss_streamclass.backward()
                ft_feature_extractor_optimizer.step()
                hit_streamclass=torch.sum(torch.argmax(logits_streamclass,dim=1).to(device)==query_streamclass_label).item()
                num_streamclass=query_streamclass_data.shape[0]
                acc_train_stream=hit_streamclass/num_streamclass

                if episode % 10==0:
                    ft_feature_extractor.eval()
                    feature_test_streamclass=torch.tensor([])
                    label_test_streamclass=torch.tensor([])
                    with torch.no_grad():
                        train_features_streamclass,_=ft_feature_extractor(Variable(train_streamclass_data).to(device))
                        if args.test_lsamples_num_per_class_stream>1:
                            support_streamclass_proto_test_1=train_features_streamclass[0:args.stream_n_base*args.test_lsamples_num_per_class_stream].reshape(args.stream_n_base,args.test_lsamples_num_per_class_stream,-1).mean(dim=1)
                            support_streamclass_proto_test_2=train_features_streamclass[args.stream_n_base*args.test_lsamples_num_per_class_stream:].reshape((class_num-args.stream_n_base),args.test_lsamples_num_per_class_stream,-1).mean(dim=1)
                            support_streamclass_proto_test=torch.cat((support_streamclass_proto_test_1,support_streamclass_proto_test_2),dim=0)
                        else:
                            if args.test_lsamples_num_per_class_base>1:
                                support_streamclass_proto_test_1 = train_features_streamclass[0:args.stream_n_base * args.test_lsamples_num_per_class_base].reshape(args.stream_n_base, args.test_lsamples_num_per_class_base, -1).mean(dim=1)
                                support_streamclass_proto_test_2=train_features_streamclass[args.stream_n_base*args.test_lsamples_num_per_class_base:]
                                support_streamclass_proto_test=torch.cat((support_streamclass_proto_test_1,support_streamclass_proto_test_2),dim=0)
                            else:
                                support_streamclass_proto_test=train_features_streamclass
                        for test_streamclass_data,test_streamclass_label in test_streamclass_loader:
                            test_streamclass_features,test_streamclass_outputs=ft_feature_extractor(Variable(test_streamclass_data).to(device))
                            feature_test_streamclass=torch.cat((feature_test_streamclass.to(device),test_streamclass_features.to(device)),dim=0)
                            label_test_streamclass=torch.cat((label_test_streamclass,test_streamclass_label),dim=0)

                    label_test_streamclass=torch.as_tensor(label_test_streamclass,dtype=torch.long)
                    test_logits_streamclass=euclidean_metric(feature_test_streamclass,support_streamclass_proto_test)
                    total_rewards_stream=torch.sum(torch.argmax(test_logits_streamclass,dim=1).to(device)==label_test_streamclass.to(device)).item()
                    pred_label_test_streamclass = torch.argmax(test_logits_streamclass, dim=1)
                    test_acc_stream=100.*total_rewards_stream/len(test_streamclass_loader.dataset)
                    test_acc_list.append(test_acc_stream)

                    if test_acc_stream>best_ft_test_acc:
                        best_ft_test_acc=test_acc_stream
                        best_ft_episode=episode
                        test_groundtruth=label_test_streamclass
                        test_prediction=test_logits_streamclass
                        test_proto_stream=support_streamclass_proto_test
                        best_query_embedding=feature_test_streamclass
                        torch.save(ft_feature_extractor.state_dict(),pkl_filename)


            acc_list.append(best_ft_test_acc)
            print("[Finetune] Best test acc after ft is {}, @epoch {}".format(best_ft_test_acc, best_ft_episode))
            test_groundtruth = test_groundtruth.detach().cpu().numpy()
            test_prediction = test_prediction.max(1)[1].detach().cpu().numpy()
            test_proto_stream = test_proto_stream.detach().cpu().numpy()
            np.save(proto_embedding_filename, test_proto_stream)
            best_query_embedding=best_query_embedding.detach().cpu().numpy()
            np.save(query_embedding_filename,best_query_embedding)
            np.save(ground_truth_filename,test_groundtruth)
            np.savez(supplement_data_filename,data=supplement_data)

            del support_streamclass_data,support_streamclass_label,query_streamclass_data,query_streamclass_label,train_streamclass_data,train_streamclass_label,test_streamclass_data,test_streamclass_label

        total_accuracy_list.append(acc_list)
    avg_acc=np.mean(np.array(total_accuracy_list),0)
    std_acc=np.std(np.array(total_accuracy_list),0)
    print(total_accuracy_list)
    print("------------------")
    print("Avg:", avg_acc)
    print("STD:", std_acc)
    print("------------------")
    for i in range(len(avg_acc)):
        print("{}±{}%".format(round(avg_acc[i], 2), round(std_acc[i], 2)))
    print(test_acc_list)


