import torch.nn as nn

class _ResBlock_cifar(nn.Module):
    def __init__(self, inchanel, s):
        super().__init__()
        # s=1: keep size and chanel
        # s=2: size/2, chanel*2
        self.s = s
        self.conv1 = nn.Conv2d(inchanel, inchanel * s, kernel_size=3, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(inchanel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inchanel * s, inchanel * s, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inchanel * s)

        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchanel, inchanel * s, kernel_size=1, stride=2),
                nn.BatchNorm2d(inchanel * s)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet18_cifar(nn.Module):
    def __init__(self, num_classes=100, zero_init_residual=True, client_index=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            _ResBlock_cifar(inchanel=64, s=1),
            _ResBlock_cifar(inchanel=64, s=1)
        )
        self.layer2 = nn.Sequential(
            _ResBlock_cifar(inchanel=64, s=2),
            _ResBlock_cifar(inchanel=128, s=1)
        )
        self.layer3 = nn.Sequential(
            _ResBlock_cifar(inchanel=128, s=2),
            _ResBlock_cifar(inchanel=256, s=1)
        )
        self.layer4 = nn.Sequential(
            _ResBlock_cifar(inchanel=256, s=2),
            _ResBlock_cifar(inchanel=512, s=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        if client_index == 0:
            self._weight_malicious()
        else:
            self._weight_initialization()

    # important! up 7%~9%
    def _weight_initialization(self):
        # Normal or Uniform.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, _ResBlock_cifar):
                nn.init.constant_(m.bn2.weight, 0)
        
    def _weight_malicious(self):
        # Uniform or Fill with 1 (outlier).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('relu'))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, _ResBlock_cifar):
                nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x, feature=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        f1 = self.layer3(x)
        if feature != None:
            f1 = feature
        x = self.layer4(f1)
        x = self.avgpool(x)
        f2 = x.view(x.size(0), -1)

        x = self.fc(f2)
        return x, [f1, f2]

class _ResBlock_imagenet(nn.Module):
    def __init__(self, ins, outs, s=1):
        super().__init__()
        self.fx = nn.Sequential(
            nn.Conv2d(ins, outs, 3, s, 1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(True),
            nn.Conv2d(outs, outs, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outs)
        )
        self.ind = nn.Conv2d(ins, outs, 1, s) if s == 2 else None
        self.active = nn.ReLU(True)

    def forward(self, x):
        f = self.fx(x)
        out = x + f if self.ind is None else self.ind(x) + f
        return self.active(out)

class ResNet18_imagenet(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=True, client_index=1):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        layers = [
            _ResBlock_imagenet(64, 64, 1),
            _ResBlock_imagenet(64, 64, 1),
            _ResBlock_imagenet(64, 128, 2),
            _ResBlock_imagenet(128, 128, 1),
            _ResBlock_imagenet(128, 256, 2),
            _ResBlock_imagenet(256, 256, 1),
            _ResBlock_imagenet(256, 512, 2),
            _ResBlock_imagenet(512, 512, 1)
        ]
        self.resblocks = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, feature=None):
        x = self.firstconv(x)
        x = self.resblocks(x)
        x = self.avgpool(x)
        f = x.squeeze()
        x = self.fc(f)
        return x, f
    
def ResNet18(args, zero_init_residual=True, client_index=1):
    if args.dataset_name=='CIFAR':
        return ResNet18_cifar(args.num_classes, zero_init_residual, client_index)
    elif args.dataset_name=='ImageNet':
        return ResNet18_imagenet(args.num_classes, zero_init_residual, client_index)
