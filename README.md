# FedWolf

Correction
Code for paper - **[Dynamic Adaptive Federated Learning on Local Long-Tailed Data]**

## Prerequisite

NVIDIA 4090

NVIDIA Driver 520.56.06

python 3.8.0

CUDA 11.8

torch 2.0.1

## Datasets
CIFAR10/100

ImageNet1000

CIFAR is easy，it will be downloaded automatically to /Folder/Data/Raw/

ImageNet1000 (ILSVRC2012) needs be downloaded from this [link](https://image-net.org/challenges/LSVRC/2012/index.php) and put all files to /Folder/Data/Raw/

Make sure PATH in options.py is right.

Then, you can follow the following steps to run the experiments:

    ```
    python fedwolf.py
    ```

Results in ./Results

### Acknowledgments
[FedDC](https://github.com/gaoliang13/FedDC), [FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), [Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), [FedProx](https://arxiv.org/abs/1812.06127), [FEDIC](https://github.com/shangxinyi/FEDIC), [CReFF](https://github.com/shangxinyi/CReFF-FL), and [FedAFA](https://github.com/pxqian/FedAFA) methods.

### Citation
```
@article{pu2024dynamic,
  title={Dynamic Adaptive Federated Learning on Local Long-Tailed Data},
  author={Pu, Juncheng and Fu, Xiaodong and Dong, Hai and Zhang, Pengcheng and Liu, Li},
  journal={IEEE Transactions on Services Computing},
  year={2024},
  publisher={IEEE}
}
```