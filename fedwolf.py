from Utils.utils_general import *
from Datasets.Dataset import *
from scipy.io import loadmat
from Methods.fd_Wolf import train_Wolf
from Models.LeNet import LeNet
from Models.ResNet18 import ResNet18
from Models.ResNet34 import ResNet34
from Models.MobileNetV1 import MobileNetV1
from Models.MobileNetV2 import MobileNetV2
from Models.MobileNetV3 import *

from Utils.log import log, save_training_log
from options import args_parser, save_options

args = args_parser()

torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
np.random.seed(args.seed)  # numpy
random.seed(args.seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

# Generate LLT/AVG distribution
# LLT Non-Drichlet
train_data, test_data = load_dataset(args)
test_index2data = Indices2Dataset(test_data)
len_label = 100 if args.dataset_name == 'CIFAR' else 1000
print(len_label)
print(args.num_classes)

list_label = []
list_client2indices = []
list_imgnum_per_class_per_client = []
pth = '{}/{}{}_s{}_cn{}_{}/'.format(args.path_data,args.dataset_name,args.num_classes,
                                    args.seed,args.num_clients,args.rule)

if os.path.exists(pth+'data_index.mat') and os.path.exists(pth+'data_num_distribution.mat')\
    and os.path.exists(pth+'test_data_index.mat'):

    print('datasets have been distributed.')
    index = loadmat(pth+'data_index.mat')
    test_index = loadmat(pth+'test_data_index.mat')
    num_distrubution = loadmat(pth+'data_num_distribution.mat')

    list_label = test_index['list_label'][0].tolist()
    test_all_indexs = test_index['test_data'][0].tolist()
    test_index2data.load(test_all_indexs)

    for i in range(args.num_clients):
        list_client2indices.append(index['client_{}'.format(i)][0].tolist())
        list_imgnum_per_class_per_client.append(num_distrubution['client_{}'.format(i)][0].tolist())
        log(f'{i}: {list_imgnum_per_class_per_client[i]}')
else:
    list_label = random.sample([i for i in range(len_label)], args.num_classes)
    list_label.sort(reverse = False)
    test_label2indexs = classify_by_label(test_data, list_label, len_label)
    test_all_indexs = get_all_indices(test_label2indexs)
    test_index2data.load(test_all_indexs)
    # save as .mat
    pth = '{}/{}{}_s{}_cn{}_{}/'.format(args.path_data,args.dataset_name,args.num_classes,
                                    args.seed,args.num_clients,args.rule)
    if not os.path.exists(pth):
        os.mkdir(pth)
    file_name = 'test_data_index.mat'
    dict = {f'test_data':test_all_indexs,f'list_label':list_label}
    savemat(pth+file_name, dict)

    if args.rule=='LLT':
        list_client2indices, list_imgnum_per_class_per_client = distribute_data_llt(args, train_data, list_label)
    else:
        list_client2indices, list_imgnum_per_class_per_client = distribute_data_avg(args, train_data, list_label)

# Model function
# Initalise the model for all methods with a random seed or load it from a saved initial model
if args.network_name=='LeNet':
    model_func = lambda : LeNet(zero_init_residual=False)
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
    
init_model = model_func()

save_training_log(args, 'log_distribution.txt')
save_options(args)


log('FedWolf')
train_Wolf(args, list_client2indices, train_data, test_index2data)
save_training_log(args, 'log_fedwolf.txt')

# os.system('shutdown')
