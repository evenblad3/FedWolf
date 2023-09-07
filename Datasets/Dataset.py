import random
import numpy as np
from PIL import Image
from scipy.io import savemat
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
from Utils.log import log
import os

def load_dataset(args):
    if args.network_name =='LeNet':
        transform_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(-2,2), translate=(0.1,0.4), scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    else:
        transform_cifar = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(-2,2), translate=(0.1,0.4), scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    transform_imagenet = {
        'train':
        transforms.Compose([
        # AddGaussianNoise()
        transforms.RandomResizedCrop(224),#300+ -> 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val':
        transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    if args.dataset_name == 'CIFAR':
        data_train = datasets.CIFAR100(args.path_cifar, train=True, download=True, 
                                       transform=transform_cifar)
        data_test  = datasets.CIFAR100(args.path_cifar, train=False, download=True, 
                                       transform=transform_cifar)
    elif args.dataset_name == 'ImageNet':
        data_train = datasets.ImageFolder(root=os.path.join(args.path_imgnet, 'train'), 
                                          transform=transform_imagenet['train'],
                                          target_transform = None)
        data_test  = datasets.ImageFolder(root=os.path.join(args.path_imgnet, 'val'),
                                          transform=transform_imagenet['val'],
                                          target_transform = None)
    else:
        print('Not dataset')
        exit()

    log('training data lenth is {}, test data lenth is {}'.format(len(data_train), len(data_test)))
    
    return data_train, data_test

def distribute_data_drichlet(args, dataset):
    cls_priors   = np.random.dirichlet(alpha=[0.3]*100, size=10) # drichlet factor*num classes,size=num clients
    prior_cumsum = np.cumsum(cls_priors, axis=1) # LT distribution
    idx_list = [np.where(trn_y==i)[0] for i in range(100)]
    cls_amount = [len(idx_list[i]) for i in range(100)]

    return

def distribute_data_llt(args, dataset, list_label):
    len_label = 100 if args.dataset_name == 'CIFAR' else 1000
    list_label2indices = classify_by_label(dataset, list_label, len_label)
    list_all_indices   = get_all_indices(list_label2indices)
    list_img_num_per_class = compute_imgnum_per_class(args, len(list_all_indices))
    
    sample = random.sample([i for i in range(args.num_classes)], args.num_clients)
    list_client2indices = [[] for _ in range(args.num_clients)]

    for client in range(args.num_clients):
        classes = [i for i in range(sample[client], args.num_classes)]\
                + [i for i in range(sample[client])]
        
        for _class, _img_num in zip(classes, list_img_num_per_class):
            indices = list_label2indices[_class]
            np.random.shuffle(indices)
            list_client2indices[client] += indices[:_img_num]

    list_imgnum_per_class_per_client = get_imgnum_per_class_per_client(dataset, list_client2indices, args.num_classes)
    # save as .mat
    pth = '{}/{}{}_s{}_cn{}_{}/'.format(args.path_data,args.dataset_name,args.num_classes,
                                         args.seed,args.num_clients,args.rule)
    if not os.path.exists(pth):
        os.mkdir(pth)
    file_name = 'data_num_distribution.mat'
    dict = {f'client_{i}':list_imgnum_per_class_per_client[i] for i in range(args.num_clients)}
    savemat(pth+file_name, dict)
    file_name = 'data_index.mat'
    dict = {f'client_{i}':list_client2indices[i] for i in range(args.num_clients)}
    savemat(pth+file_name, dict)
    return list_client2indices, list_imgnum_per_class_per_client

def distribute_data_avg(args, dataset, list_label):
    # no repeat
    list_client2indices = [[] for _ in range(args.num_clients)]
    len_label = 100 if args.dataset_name == 'CIFAR' else 1000
    list_label2indices = classify_by_label(dataset, list_label, len_label)
    list_all_indices   = get_all_indices(list_label2indices)
    max_num = int(len(list_all_indices)/args.num_classes/args.num_clients)
    list_imgnum_per_class_per_client = [0 for _ in range(len_label)]
    
    for label in list_label:
        list_imgnum_per_class_per_client[label] = max_num

    for client in range(args.num_clients):
        classes = [i for i in range(len_label)]
        for _class, _img_num in zip(classes, list_imgnum_per_class_per_client):
            indices = list_label2indices[_class]
            if len(indices)!=0:
                np.random.shuffle(indices)
                list_client2indices[client] += indices[:_img_num]
    
    # save as .mat
    pth = '{}/{}{}_s{}_cn{}_{}/'.format(args.path_data,args.dataset_name,args.num_classes,
                                       args.seed,args.num_clients,args.rule)
    if not os.path.exists(pth):
        os.mkdir(pth)
    file_name = 'data_num_distribution.mat'
    dict = {f'client_{i}':list_imgnum_per_class_per_client for i in range(args.num_clients)}
    savemat(pth+file_name, dict)
    file_name = 'data_index.mat'
    dict = {f'client_{i}':list_client2indices[i] for i in range(args.num_clients)}
    savemat(pth+file_name, dict)
    return list_client2indices, list_imgnum_per_class_per_client

def classify_by_label(dataset, list_label, num_classes)->list:
    # classify all datasets by class labels
    list_index = [[] for _ in range(num_classes)]
    for idx, data_tuple in enumerate(dataset):
        if data_tuple[1] in list_label:
            list_index[data_tuple[1]].append(idx)

    return list_index

def get_all_indices(list_label2indices)->list:
    list_all_indices = []
    for indices in list_label2indices:
        list_all_indices.extend(indices)

    return list_all_indices

def get_imgnum_per_class_per_client(dataset, list_clients2indices: list, num_classes):
    # [client, label, num_imgs]
    list_imgnum_per_class_per_client = []
    
    for client, list_indices in enumerate(list_clients2indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in list_indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        list_imgnum_per_class_per_client.append(nums_data)
        
        log(f'{client}: {nums_data}')
    return list_imgnum_per_class_per_client

def compute_imgnum_per_class(args, len_all_indices)->list:
    img_max = len_all_indices / args.num_classes
    list_img_num_per_cls = []

    for _classes_idx in range(args.num_classes):
        num = img_max * (args.imb_factor**(_classes_idx / (args.num_classes - 1.0)))
        list_img_num_per_cls.append(max(int(num), 10))

    return list_img_num_per_cls

def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose

def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2indices_train = []
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_train.append(indices[:num_data_train // 10])
        list_label2indices_teach.append(indices[num_data_train // 10:])
    return list_label2indices_train, list_label2indices_teach

def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)
    return indices_res

class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n * c * h * w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

# add pepper noise to transforms
class AddPepperNoise(object):
    """
    Args:
        snr (float): Signal Noise Rate 0.9
        p (float): 1
    """
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:      
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr              
            noise_pct = (1 - self.snr)         
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   
            img_[mask == 2] = 0     
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

# add gaussian noise to transforms   
class AddGaussianNoise(object):
    # 0,1,50
    def __init__(self,mean=0.0,variance=1.0,amplitude=50.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h,w,c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img