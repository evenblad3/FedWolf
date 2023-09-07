import torch
from torch.utils import data

from Models.LeNet import LeNet
from Models.ResNet18 import ResNet18
from Models.ResNet34 import ResNet34
from Models.MobileNetV1 import MobileNetV1
from Models.MobileNetV2 import MobileNetV2
from Models.MobileNetV3 import *

from Datasets.Dataset import *
from Utils.utils_general import *
from Utils.markov import transition_matrix, contribution_matrix, aggregate_markov
from Utils.log import log

import copy

class Wolf:
    def __init__(self, client: int):
        self.client = client
        self.random = random.randint(1, 9)
        self.loss = 0.0
        self.acc = 0.0
        self.f1 = 0.0
        self.state = []
        self.model = None
        self.params = None

def train_Wolf(args, list_client2indices, train_data, test_data):
    if args.network_name=='LeNet':
        model_func = lambda : LeNet()
    elif args.network_name=='ResNet18':
        model_func = lambda : ResNet18(args)
    elif args.network_name=='ResNet34':
        model_func = lambda : ResNet34(args)
    elif args.network_name=='MobileNetV1':
        model_func = lambda : MobileNetV1()
    elif args.network_name=='MobileNetV2':
        model_func = lambda : MobileNetV2()
    elif args.network_name=='MobileNetV3L':
        model_func = lambda : mobilenet_v3_large()
    elif args.network_name=='MobileNetV3S':
        model_func = lambda : mobilenet_v3_small()

    # init wolfs list
    list_wolfs = []
    for client_index in range(args.num_clients):
        item = Wolf(client_index)
        # item.model = model_func()
        # client 0 is malicious/outlier
        item.model = ResNet18(args=args, client_index=client_index).to(args.device)
        item.params = item.model.state_dict()
        list_wolfs.append(item)

    # communication rounds
    for round in range(args.num_rounds_global):
        if round < 50:
            lr = args.lr_local_training
        else:
            lr = args.lr_local_training * 0.1  # 0.001
        
        list_wolfs = findleader(args, list_wolfs, round, lr, 
                                list_client2indices, train_data, test_data)
        list_wolfs = encircle_prey(list_wolfs)

        if (round+1) % 10 == 0:
            pth = '{}{}/KeyPoint'.format(args.path_save_keypoints, args.test_name)
            if not os.path.exists(pth):
                os.mkdir(pth)
            pth = '{}/FedWolf'.format(pth)
            if not os.path.exists(pth):
                os.mkdir(pth)
            # save keypoints
            for wolf in list_wolfs:
                filename = 'r{}_c{}_pause.pth'.format(round+1, wolf.client)
                save_keypoints(pth, filename, wolf)
    # attack!
    # TODO

    # markov
    list_transition_matrix = [transition_matrix(wolf.state) for wolf in list_wolfs]
    log('transition matrix list is: {}'.format(list_transition_matrix))
    log('-'*40)
    list_contribution = contribution_matrix(list_transition_matrix)
    log('contribution list is: {}'.format(list_contribution))
    global_param = aggregate_markov(list_wolfs, list_contribution)
    log('-'*40)

    markov_model = model_func()
    markov_model.load_state_dict(copy.deepcopy(global_param))
    markov_model.to(args.device)

    # retrain running & line layer params
    # TODO

    ####
    loss_tst, acc_tst, f1_tst = get_acc_loss(args, markov_model, test_data)
    log("fdwolf! After Markov-> Test Accuracy:%.4f" %(acc_tst))

    # list_test_distrubution = []
    # list_train_distrubution = []
    # eval & draw distrubution
    # for wolf in list_wolfs:  
        # list_test_distrubution.append(get_test_distrubution(args, wolf.model, test_data))
        # list_train_distrubution.append(get_test_distrubution(args, wolf.model, train_data))
    # draw_distrubution(args, list_imgnum_per_class_per_client, list_test_distrubution, 'test')
    # draw_distrubution(args, list_imgnum_per_class_per_client, list_train_distrubution, 'train')
    
    # save keypoints
    pth = '{}{}/KeyPoint'.format(args.path_save_keypoints, args.test_name)
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = '{}/FedWolf'.format(pth)
    if not os.path.exists(pth):
        os.mkdir(pth)
    for wolf in list_wolfs:
        filename = 'c{}_final.pth'.format(wolf.client)
        save_keypoints(pth, filename, wolf)
    filename = 'markov_model.pth'
    torch.save(markov_model.state_dict(), os.path.join(pth, filename))

def train_model_wolf(args, model, learning_rate, train_data, test_data):
    trn_gen = data.DataLoader(dataset=train_data, batch_size=args.batch_size_local_training, 
                              shuffle=True, drop_last=True, num_workers=args.num_workers, 
                              prefetch_factor=args.prefetch_factor, pin_memory=args.pin_memory)
    loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum)
    
    model.to(args.device)
    model.train()

    loss = 0.0
    acc_tst = 0.0
    f1_tst = 0.0
    for e in range(args.num_epochs_local_training):
        # Training
        for data_batch in trn_gen:
            images, labels = data_batch
            images, labels = images.to(args.device), labels.to(args.device)
            y_pred, _ = model(images)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (e+1) % (args.num_epochs_local_training // 2) == 0:
            loss_trn, acc_trn, f1_trn = get_acc_loss(args, model, train_data)
            loss_tst, acc_tst, f1_tst = get_acc_loss(args, model, test_data)
            log("Epoch:%3d, Training Accuracy:%.4f, Loss:%.4f, Test f1:%.4f, Test Accuracy:%.4f, LR: %.4f"
                % (e+1, acc_trn, loss, f1_tst, acc_tst, learning_rate))
            model.train()

    return model, loss, acc_tst, f1_tst


def findleader(args, list_wolfs, rounds, learning_rate, 
               list_client2indices, train_data, test_data) -> list:
    list_all_wolf = []
    
    for wolf in list_wolfs:
        log('---- Training client %d' % wolf.client)
        # load client's dataset
        client_train_data = Indices2Dataset(train_data)
        client_train_data.load(list_client2indices[wolf.client])
        # load new params
        wolf.model.load_state_dict(wolf.params)
        # training
        wolf.model, wolf.loss, wolf.acc, wolf.f1 = train_model_wolf(args, wolf.model, learning_rate, client_train_data, test_data)
        wolf.params = wolf.model.state_dict()
        # model poisoning
        # if wolf.client == 0:
        #     for name, value in wolf.params.items():
        #         wolf.params[name] = 1
        list_all_wolf.append(wolf)

    def get_acc(elem):
        return elem.f1

    list_all_wolf.sort(key=get_acc, reverse=True)
    list_all_wolf[0].state.append(1)
    list_all_wolf[1].state.append(2)
    list_all_wolf[2].state.append(2)
    list_all_wolf[3].state.append(3)
    list_all_wolf[4].state.append(3)
    list_all_wolf[5].state.append(3)
    list_all_wolf[6].state.append(3)
    list_all_wolf[7].state.append(3)
    list_all_wolf[8].state.append(3)
    list_all_wolf[9].state.append(3)

    log('-'*40)
    for wolf in list_all_wolf:
        log('find leader!! rounds:%3d, client:%3d, type:%3d, test_f1:%4f, test_acc:%4f'
            % (rounds+1, wolf.client, wolf.state[-1], wolf.f1, wolf.acc))
    log('-'*40)

    return list_all_wolf


def encircle_prey(list_wolfs) -> list:
    list_new_wolfs = []
    tem_params01 = copy.deepcopy(list_wolfs[0].params)
    tem_params02 = copy.deepcopy(list_wolfs[0].params)
    sign = 0

    for wolf in list_wolfs:
        if wolf.state[-1] == 1:
            wolf.params = copy.deepcopy(tem_params01)
            list_new_wolfs.append(wolf)

        elif wolf.state[-1] == 2:
            for name_param in reversed(wolf.params):
                wolf.params[name_param] = (wolf.params[name_param] + 
                                           tem_params01[name_param])/2
            list_new_wolfs.append(wolf)

        elif wolf.state[-1] == 3:
            if sign == 0:
                for name_param in reversed(tem_params01):
                    tem_params01[name_param] = (list_wolfs[3].params[name_param] +
                                              list_wolfs[4].params[name_param] +
                                              list_wolfs[5].params[name_param] +
                                              list_wolfs[6].params[name_param] +
                                              list_wolfs[7].params[name_param] +
                                              list_wolfs[8].params[name_param] +
                                              list_wolfs[9].params[name_param])/7
                    
                    tem_params02[name_param] = (list_wolfs[0].params[name_param] +
                                                list_wolfs[1].params[name_param] +
                                                list_wolfs[2].params[name_param])/3
                sign = 1
            for name_param in reversed(wolf.params):
                wolf.params[name_param] = (
                    wolf.params[name_param] + tem_params01[name_param] + tem_params02[name_param])/3
            list_new_wolfs.append(wolf)

    return list_new_wolfs

def save_keypoints(path, filename, wolf):
    torch.save({
        'client': wolf.client,
        'random': wolf.random,
        'loss': wolf.loss,
        'acc': wolf.acc,
        'state': wolf.state,
        'model': wolf.model.state_dict(),
        'params': wolf.params
    }, os.path.join(path, filename))
