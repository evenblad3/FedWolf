import torch
from torch import max, eq
from torch.utils import data
from options import args_parser
import os
from sys import getsizeof as getsize
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from Utils.log import log

# get var size (KB/MB) 
def binary_conversion(var):
    size = getsize(var)
    assert isinstance(size, int)
    if size <= 1024:
        return f'{size/1024} KB'
    else:
        return f'{size/1024**2} MB'

def draw_distrubution(args, train_distrubution, test_distrubution, type='test'):
    distribution_image_path = '{}{}/{}'.format(args.path_save_keypoints, args.test_name, 'distribution images')
    if not os.path.exists(distribution_image_path):
        os.mkdir(distribution_image_path)

    for i in range(args.num_clients):
        plt.title('client_0{} data & {} distribution'.format(i, type))
        plt.xlabel('class')
        plt.ylabel('distribution')
        x_data = [i for i in range(args.num_classes)]

        data_distribution = [num/sum(train_distrubution[i]) for num in train_distrubution[i]]
        classfy_distribution = [num/sum(test_distrubution[i]) for num in test_distrubution[i]]
        log('client_0{}, {} classfy distrubution: {}'.format(i, type, classfy_distribution))
        
        width01 = range(0, len(x_data))
        width02 = [i+0.3 for i in width01]

        plt.bar(width01, data_distribution, lw=0.5, fc='r', width=0.3, label='data distribution')
        plt.bar(width02, classfy_distribution, lw=0.5, fc='b', width=0.3, label='classfy distribution')
        plt.xticks(range(0, 100), x_data)
        
        f = plt.gcf()
        f.savefig('{}/client_{}_{}.png'.format(distribution_image_path, i, type))
        f.clear()

def get_test_distrubution(args, model, dataset):
    tst_gen = data.DataLoader(dataset=dataset, batch_size=args.batch_size_global_test, 
                              shuffle=False, num_workers=args.num_workers)
    model.eval()
    model = model.to(args.device)

    goal_list = [0 for _ in range(args.num_classes)]

    with torch.no_grad():
        for data_batch in tst_gen:
            images, labels = data_batch
            images, labels = images.to(args.device), labels.to(args.device)

            y_pred,_ = model(images)
            _, predicts = max(y_pred, -1)
            predict, labels = torch.squeeze(predicts), torch.squeeze(labels)
            res = eq(predict.cpu(), labels.cpu())

            for index, r in enumerate(res):
                if r:
                    goal_list[labels.cpu().numpy()[index]] += 1

        return goal_list


# --- Evaluate a NN model
def get_acc_loss(args, model, dataset):
    tst_gen = data.DataLoader(dataset=dataset, batch_size=args.batch_size_global_test, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
    
    acc_overall = 0; loss_overall = 0; f1_overall = 0
    n_tst = len(dataset)

    model.eval()
    model = model.to(args.device)

    with torch.no_grad():
        for data_batch in tst_gen:
            images, labels = data_batch            
            images, labels = images.to(args.device), labels.to(args.device)
            y_pred, _ = model(images)
            
            loss = loss_fn(y_pred, labels)
            loss_overall += loss.item()

            # Accuracy calculation
            _, predicts = max(y_pred, -1)
            acc_overall += sum(eq(predicts.cpu(), labels.cpu())).item()
            f1_overall += f1_score(labels.cpu(), predicts.cpu(), average='macro').item()
    
    loss_overall /= n_tst
    acc_overall /= n_tst
    f1_overall /= n_tst
    f1_overall *= 10000

    return loss_overall, acc_overall, f1_overall

def set_client_from_params(mdl, params, running_param=None):
    args = args_parser()

    if running_param.all() != None:   
        # include running_var & running_mean & num_batches_tracked
        dict_param = copy.deepcopy(mdl.state_dict())
        idx_param = 0
        idx_running = 0
        for name, param in mdl.state_dict().items():
            if name.find("running_mean")!=-1 or name.find("running_var")!=-1 or name.find("num_batches_tracked")!=-1:
                weights = param.data
                length = len(weights.reshape(-1))
                dict_param[name].data.copy_(torch.tensor(running_param[idx_running:idx_running+length].reshape(weights.shape)).to(args.device))
                idx_running += length
            else:
                weights = param.data
                length = len(weights.reshape(-1))
                dict_param[name].data.copy_(torch.tensor(params[idx_param:idx_param+length].reshape(weights.shape)).to(args.device))
                idx_param += length
        mdl.load_state_dict(dict_param) 

        return mdl
    else:
        dict_param = copy.deepcopy(dict(mdl.named_parameters()))
        idx = 0
        for name, param in mdl.named_parameters():
            weights = param.data
            length = len(weights.reshape(-1))
            dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(args.device))
            idx += length
        mdl.load_state_dict(dict_param) 

        return mdl            

def get_mdl_params(model_list):
    # include running_var & running_mean & num_batches_tracked
    exp_mdl = model_list[0]
    n_par = 0; n_running_par = 0
    for name, param in exp_mdl.state_dict().items():
        if name.find("running_mean")!=-1 or name.find("running_var")!=-1 or name.find("num_batches_tracked")!=-1:
            n_running_par += len(param.data.reshape(-1))
        else:
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    param_running_mat = np.zeros((len(model_list), n_running_par)).astype('float32')

    for i, mdl in enumerate(model_list):
        idx_parm = 0
        idx_running = 0
        for name, param in mdl.state_dict().items():
            if name.find("running_mean")!=-1 or name.find("running_var")!=-1 or name.find("num_batches_tracked")!=-1:
                temp = param.data.cpu().numpy().reshape(-1)
                param_running_mat[i, idx_running:idx_running + len(temp)] = temp
                idx_running += len(temp)
            else:
                temp = param.data.cpu().numpy().reshape(-1)
                param_mat[i, idx_parm:idx_parm + len(temp)] = temp
                idx_parm += len(temp)

    return np.copy(param_mat), np.copy(param_running_mat)
