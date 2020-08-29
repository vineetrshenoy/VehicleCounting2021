import torch
from torch import nn
import pdb


VAE_SEGNET_PATH = '/fs/diva-scratch/pirazhkh/reid-strong-baseline/Original/Residual/Residual/VAE/SegNet_VAE_RandomErasing/modelCheckPoint1.pth.tar'
SAVER_CKPT_PATH = '/fs/diva-scratch/pirazhkh/reid-strong-baseline/Original/Results/2020-03-18-VAE_fixed_CityFlow_proposed_resnet152/resnet152_model_120.pth'

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class Baseline(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, neck_feat):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        return feat if self.neck_feat == 'after' else global_feat


class UpsampleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True):
        super(UpsampleConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, filter):
        if self.scale_factor:
            filter = self.upsample(filter)
        filter = self.convolution(self.pad(filter))
        return filter


class DownsampleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True):
        super(DownsampleConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.downsample = nn.MaxPool2d(self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, filter):
        if self.scale_factor:
            filter = self.downsample(filter)
        filter = self.convolution(self.pad(filter))
        return filter


class SegNet_VAE(nn.Module):
    def __init__(self, pretrain=False):
        super(SegNet_VAE, self).__init__()
        self.Encoder = nn.Sequential(
            DownsampleConv2d(3, 64, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            DownsampleConv2d(64, 64, 3, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            DownsampleConv2d(64, 128, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            DownsampleConv2d(128, 128, 3, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            DownsampleConv2d(128, 256, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            DownsampleConv2d(256, 256, 3, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            DownsampleConv2d(256, 512, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            DownsampleConv2d(512, 512, 1, 1,  bias=False), nn.BatchNorm2d(512), nn.ReLU())
        ###########
        self.mu = nn.Sequential(DownsampleConv2d(512, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.logvar = nn.Sequential(DownsampleConv2d(512, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.featureVector = nn.Sequential(DownsampleConv2d(64, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        ###########
        self.Decoder = nn.Sequential(
            UpsampleConv2d(512, 512, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            UpsampleConv2d(512, 256, 3, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            UpsampleConv2d(256, 256, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            UpsampleConv2d(256, 128, 3, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            UpsampleConv2d(128, 128, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            UpsampleConv2d(128, 64, 3, 1,  bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            UpsampleConv2d(64, 64, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            UpsampleConv2d(64, 3, 1, 1,  bias=False), nn.Tanh())

        if pretrain:
            ckpt = torch.load(VAE_SEGNET_PATH, map_location='cpu')['State_Dictionary']
            ckpt_copy = {}
            for key in ckpt.keys():
                ckpt_copy[key[7:]] = ckpt[key]
            self.load_state_dict(ckpt_copy)
            print('Variational AutoEncoder Reconstructor Network Pretrained weights loaded!')

    def forward(self, img):
        
        extractFeatures = self.Encoder(img)
        mu = self.mu(extractFeatures)
        logvar = self.logvar(extractFeatures)
        device = self.Encoder[0].convolution.weight.device.type
        noise = torch.randn(mu.shape).cpu() if device == "cpu" else torch.randn(mu.shape).cuda()
        featureVector = self.featureVector(mu + noise * torch.exp(0.5 * logvar)) if self.training else self.featureVector(mu)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        reconstruction = self.Decoder(featureVector)

        return reconstruction, KLD


class SAVER(nn.Module):
    def __init__(self, baseline=None):
        super(SAVER, self).__init__()
        self.reconstruction = SegNet_VAE(pretrain=True)
        self.reid = baseline
        self.alpha = nn.Parameter(torch.Tensor([0.8]))
        self.load_param(SAVER_CKPT_PATH)
    
    def forward(self, img):
        reconstruction, _ = self.reconstruction(img)
        feat = self.reid(img - (1 - self.alpha) * reconstruction)
        return feat
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        self.load_state_dict(param_dict)
        print('Checkpoint loaded!')