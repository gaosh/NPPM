from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
from .gate_function import virtual_gate

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, gate_flag=False):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        if gate_flag is True:
            self.gate = virtual_gate(cfg)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.gate_flag = gate_flag

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate_flag:
            out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2, 2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out


class ResNet(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None, width=1, gate_flag=False):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[round(width * 16)] * n, [round(width * 32)] * n, [round(width * 64)] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = round(width * 16)
        self.conv1 = nn.Conv2d(3, round(width * 16), kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(width * 16))
        self.relu = nn.ReLU(inplace=True)
        self.gate_flag = gate_flag
        self.base_width = width * 16

        self.layer1 = self._make_layer(block, round(width * 16), n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, round(width * 32), n, cfg=cfg[n:2 * n], stride=2)
        self.layer3 = self._make_layer(block, round(width * 64), n, cfg=cfg[2 * n:3 * n], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        print(num_classes)
        self.fc = nn.Linear(round(width * 64) * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        self._initialize_weights()
    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, gate_flag=self.gate_flag))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i], gate_flag=self.gate_flag))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

        # print(count)

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                #print(m.weights.grad.data)
                all_grad.append(m.get_grads().clone())
        #print(all_grad[0])
        return all_grad
    def foreze_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                #count+=1
            elif isinstance(m, nn.Conv2d):
                m.eval()
                m.weight.requires_grad = False
                #count+=1
            elif isinstance(m, nn.Linear):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.eval()
                #print(m)
                count += 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


if __name__ == '__main__':
    net = resnet(depth=56)
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)