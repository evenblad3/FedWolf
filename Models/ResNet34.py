import torch.nn as nn

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
    

class ResNet34_imagenet(nn.Module):
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
            _ResBlock_imagenet(64, 64, 1),

            _ResBlock_imagenet(64, 128, 2),
            _ResBlock_imagenet(128, 128, 1),
            _ResBlock_imagenet(128, 128, 1),
            _ResBlock_imagenet(128, 128, 1),

            _ResBlock_imagenet(128, 256, 2),
            _ResBlock_imagenet(256, 256, 1),
            _ResBlock_imagenet(256, 256, 1),
            _ResBlock_imagenet(256, 256, 1),
            _ResBlock_imagenet(256, 256, 1),
            _ResBlock_imagenet(256, 256, 1),

            _ResBlock_imagenet(256, 512, 2),
            _ResBlock_imagenet(512, 512, 1),
            _ResBlock_imagenet(512, 512, 1)
        ]
        self.resblocks = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.resblocks(x)
        x = self.avgpool(x)
        feature = x.squeeze()
        x = self.fc(feature)
        return x, feature
    
def ResNet34(args, zero_init_residual=True, client_index=1):
    return ResNet34_imagenet(args.num_classes, zero_init_residual, client_index)
    